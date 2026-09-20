from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shlex
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml


PROXY_ENV_VARS = (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
)
FINAL_STATUSES = {
    "success",
    "invalid_input",
    "model_failed",
    "parse_failed",
    "metric_failed",
    "skipped_by_policy",
}
MINI_MODEL_CLASS = "benchmark.swe_bench_lite.model.SparseVLLMLitellmModel"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"hf_[A-Za-z0-9]{16,}"),
    re.compile(r"AIza[A-Za-z0-9_-]{16,}"),
    re.compile(r"(?i)Bearer\s+[A-Za-z0-9._~+/-]{12,}=*"),
)
SENSITIVE_KEY_PATTERN = re.compile(
    r"(?i)(?:^|_)(?:api_?key|api_?token|access_?token|auth_?token|bearer_?token|"
    r"refresh_?token|id_?token|secret|client_?secret|password|credential|credentials|"
    r"authorization)(?:$|_)|^token$"
)
INLINE_SECRET_PATTERN = re.compile(
    r"(?i)(?:--?(?:api[-_]?key|access[-_]?token|auth[-_]?token|token|password|secret)"
    r"\s+|(?:api[-_]?key|access[-_]?token|auth[-_]?token|token|password|secret)=)\S+"
)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class RunnerError(RuntimeError):
    """Raised when an evaluation boundary cannot be validated."""


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RunnerError(f"Required JSON file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RunnerError(f"Invalid JSON in {path}: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(payload.encode("utf-8"))


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_state(repo: Path, *, source_roots: Sequence[str]) -> dict[str, Any]:
    if not (repo / ".git").exists():
        return {
            "path": str(repo),
            "commit": None,
            "source_dirty": None,
            "tracked_diff_sha256": None,
            "untracked_source_files": [],
        }

    def run(*args: str) -> str:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            text=True,
            capture_output=True,
        )
        return proc.stdout.strip()

    diff_proc = subprocess.run(
        ["git", "-C", str(repo), "diff", "--binary", "HEAD", "--"],
        check=True,
        capture_output=True,
    )
    untracked_output = run(
        "ls-files", "--others", "--exclude-standard", "--", *source_roots
    )
    untracked_files = []
    for relative in untracked_output.splitlines():
        path = repo / relative
        if path.is_file():
            untracked_files.append(
                {"path": relative, "sha256": _sha256_file(path)}
            )

    tracked_diff_sha256 = _sha256_bytes(diff_proc.stdout)
    source_dirty = bool(diff_proc.stdout or untracked_files)
    return {
        "path": str(repo.resolve()),
        "commit": run("rev-parse", "HEAD"),
        "source_dirty": source_dirty,
        "tracked_diff_sha256": tracked_diff_sha256,
        "untracked_source_files": untracked_files,
    }


def assert_runtime_provenance_matches(
    expected: dict[str, Any], current: dict[str, Any]
) -> None:
    if expected != current:
        raise RunnerError(
            "Runtime provenance drift detected between stages. Use a new --run-dir "
            "after changing adapter code, SWE-bench code, Python, or package versions."
        )


def build_official_run_id(run_id: str, predictions_sha256: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", predictions_sha256) is None:
        raise RunnerError("Prediction hash must be a lowercase SHA-256 digest")
    return f"{run_id}-pred-{predictions_sha256[:12]}"


def _reject_secrets(value: Any, *, source: Path, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            key_text = str(key)
            normalized_key = key_text.replace("-", "_").lower()
            if (
                not normalized_key.endswith("_env")
                and SENSITIVE_KEY_PATTERN.search(normalized_key)
                and nested not in (None, "")
            ):
                raise RunnerError(
                    f"Potential secret field {path}.{key_text} in {source}; "
                    "store credentials only in environment variables"
                )
            _reject_secrets(nested, source=source, path=f"{path}.{key_text}")
        return
    if isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_secrets(nested, source=source, path=f"{path}[{index}]")
        return
    if not isinstance(value, str):
        return

    if any(pattern.search(value) for pattern in SECRET_PATTERNS):
        raise RunnerError(f"Potential secret value at {path} in {source}")
    if INLINE_SECRET_PATTERN.search(value):
        raise RunnerError(f"Inline command secret at {path} in {source}")
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme and (parsed.username is not None or parsed.password is not None):
        raise RunnerError(f"URL credentials at {path} in {source} are secret")
    if parsed.scheme:
        for key, nested in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True):
            if SENSITIVE_KEY_PATTERN.search(key.replace("-", "_")) and nested:
                raise RunnerError(f"URL query secret at {path} in {source}")


def _load_dataset(dataset: str, split: str) -> list[dict[str, Any]]:
    try:
        from swebench.harness.utils import load_swebench_dataset
    except ImportError as exc:
        raise RunnerError(
            "swebench is not installed in this Python environment. Run the adapter "
            "with the environment that provides mini-SWE-agent and SWE-bench."
        ) from exc

    rows = list(load_swebench_dataset(dataset, split))
    if not rows:
        raise RunnerError(f"Dataset {dataset!r} split {split!r} is empty")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("instance_id"), str):
            raise RunnerError(f"Dataset row {index} has no string instance_id")
    return rows


def _parse_slice(value: str | None, length: int) -> slice:
    if not value:
        return slice(0, length)
    match = re.fullmatch(r"(\d*):(\d*)", value)
    if match is None:
        raise RunnerError("--slice must use START:STOP with non-negative integers")
    start = int(match.group(1)) if match.group(1) else 0
    stop = int(match.group(2)) if match.group(2) else length
    if start > stop or stop > length:
        raise RunnerError(f"Invalid --slice {value!r} for {length} selected instances")
    return slice(start, stop)


def select_rows(
    rows: Sequence[dict[str, Any]],
    *,
    instance_ids_file: Path | None,
    instance_filter: str | None,
    slice_spec: str | None,
) -> list[dict[str, Any]]:
    by_id = {row["instance_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise RunnerError("Dataset contains duplicate instance_id values")

    if instance_ids_file is not None:
        try:
            requested = [
                line.strip()
                for line in instance_ids_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except FileNotFoundError as exc:
            raise RunnerError(f"Instance id file does not exist: {instance_ids_file}") from exc
        if not requested:
            raise RunnerError(f"Instance id file is empty: {instance_ids_file}")
        if len(set(requested)) != len(requested):
            raise RunnerError(f"Instance id file contains duplicates: {instance_ids_file}")
        unknown = sorted(set(requested) - set(by_id))
        if unknown:
            raise RunnerError(f"Unknown instance ids: {unknown[:10]}")
        selected = [by_id[instance_id] for instance_id in requested]
    else:
        selected = list(rows)

    if instance_filter:
        try:
            pattern = re.compile(instance_filter)
        except re.error as exc:
            raise RunnerError(f"Invalid --filter regex: {exc}") from exc
        selected = [row for row in selected if pattern.search(row["instance_id"])]

    selected = selected[_parse_slice(slice_spec, len(selected))]
    if not selected:
        raise RunnerError("Instance selection is empty")
    return selected


def _instance_image_names(rows: Sequence[dict[str, Any]]) -> list[str]:
    names = []
    for row in rows:
        instance_id = row["instance_id"]
        image = f"swebench/sweb.eval.x86_64.{instance_id.lower()}:latest"
        names.append(image.replace("__", "_1776_"))
    return names


def _require_local_images(image_names: Sequence[str]) -> None:
    if not image_names:
        raise RunnerError("No Docker images were derived for the selected instances")
    try:
        daemon = subprocess.run(
            ["docker", "info"],
            text=True,
            capture_output=True,
            timeout=10,
        )
        if daemon.returncode != 0:
            detail = (daemon.stderr or daemon.stdout).strip()
            raise RunnerError(
                "Docker daemon is unavailable"
                + (f": {detail[:500]}" if detail else "")
            )
        proc = subprocess.run(
            ["docker", "image", "inspect", *image_names],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError as exc:
        raise RunnerError("docker is not installed or is not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise RunnerError("Docker daemon check timed out after 10 seconds") from exc
    if proc.returncode == 0:
        return

    missing = []
    for image in image_names:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            missing.append(image)
    preview = "\n".join(missing[:10])
    raise RunnerError(
        f"{len(missing)} selected SWE-bench Docker images are missing. The adapter "
        f"does not pull images automatically. First missing images:\n{preview}"
    )


def _validate_local_server_manifest(
    path: Path,
    *,
    api_base: str,
    served_model_name: str,
) -> dict[str, Any]:
    manifest = _read_json(path)
    if not isinstance(manifest, dict):
        raise RunnerError(f"Server manifest must be a JSON object: {path}")
    required = {
        "command",
        "model_path",
        "served_model_name",
        "cuda_visible_devices",
        "server_port",
        "engine_kwargs",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise RunnerError(f"Server manifest is missing fields: {missing}")
    for field in ("command", "model_path", "served_model_name", "cuda_visible_devices"):
        if not isinstance(manifest[field], str) or not manifest[field].strip():
            raise RunnerError(f"server manifest {field} must be a non-empty string")
    if not isinstance(manifest["engine_kwargs"], dict):
        raise RunnerError("server manifest engine_kwargs must be a JSON object")
    _reject_secrets(manifest, source=path)
    engine_kwargs = manifest["engine_kwargs"]
    if "max_model_len" not in engine_kwargs:
        raise RunnerError("server manifest engine_kwargs must record max_model_len")
    if not isinstance(engine_kwargs["max_model_len"], int) or engine_kwargs["max_model_len"] <= 0:
        raise RunnerError("server manifest max_model_len must be a positive integer")
    if "sparse_method" not in engine_kwargs:
        raise RunnerError("server manifest engine_kwargs must record sparse_method")
    if manifest["served_model_name"] != served_model_name:
        raise RunnerError(
            "server manifest served_model_name does not match --served-model-name: "
            f"{manifest['served_model_name']!r} != {served_model_name!r}"
        )
    try:
        server_port = int(manifest["server_port"])
    except (TypeError, ValueError) as exc:
        raise RunnerError("server manifest server_port must be an integer") from exc
    parsed = urllib.parse.urlparse(api_base)
    if parsed.port is not None and server_port != parsed.port:
        raise RunnerError(
            f"server manifest port {manifest['server_port']} does not match API base {api_base}"
        )
    return manifest


def _models_url(api_base: str) -> str:
    return api_base.rstrip("/") + "/models"


def check_openai_server(
    api_base: str,
    *,
    api_key: str,
    served_model_name: str,
    timeout: float,
    use_environment_proxy: bool,
) -> None:
    request = urllib.request.Request(
        _models_url(api_base),
        headers={"Authorization": f"Bearer {api_key}"},
    )
    opener = (
        urllib.request.build_opener()
        if use_environment_proxy
        else urllib.request.build_opener(urllib.request.ProxyHandler({}))
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RunnerError(
            f"OpenAI model health check failed for {_models_url(api_base)}: {exc}"
        ) from exc
    try:
        model_ids = {item["id"] for item in payload["data"]}
    except (KeyError, TypeError) as exc:
        raise RunnerError("OpenAI /models response does not contain data[].id") from exc
    if served_model_name not in model_ids:
        raise RunnerError(
            f"OpenAI server does not advertise {served_model_name!r}; available models: "
            f"{sorted(model_ids)}"
        )


def render_mini_config(
    *,
    step_limit: int,
    cost_limit: float,
    wall_time_limit_seconds: int,
    cost_tracking: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool,
    preserve_thinking: bool,
    api_base: str | None,
) -> str:
    lines = [
        "agent:",
        f"  step_limit: {step_limit}",
        f"  cost_limit: {cost_limit}",
        f"  wall_time_limit_seconds: {wall_time_limit_seconds}",
        "model:",
        f"  model_class: {MINI_MODEL_CLASS}",
        f"  cost_tracking: {json.dumps(cost_tracking)}",
        "  model_kwargs:",
        "    drop_params: true",
        "    parallel_tool_calls: true",
        f"    max_tokens: {max_tokens}",
        f"    temperature: {temperature}",
        f"    top_p: {top_p}",
        "    extra_body:",
        f"      enable_thinking: {str(enable_thinking).lower()}",
        f"      preserve_thinking: {str(preserve_thinking).lower()}",
        "      chat_template_kwargs:",
        f"        clear_thinking: {str(not preserve_thinking).lower()}",
    ]
    if api_base:
        lines.append(f"    api_base: {json.dumps(api_base)}")
    lines.extend(["environment:", "  pull_timeout: 30", ""])
    return "\n".join(lines)


def _redact(line: str) -> str:
    for pattern in SECRET_PATTERNS:
        line = pattern.sub("[redacted]", line)
    return INLINE_SECRET_PATTERN.sub("[redacted-secret-argument]", line)


def _run_logged(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(
                list(command),
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise RunnerError(f"Failed to start command {shlex.join(command)}: {exc}") from exc
        assert process.stdout is not None
        for line in process.stdout:
            safe_line = _redact(line)
            log.write(safe_line)
            log.flush()
            sys.stdout.write(safe_line)
            sys.stdout.flush()
        returncode = process.wait()
    if returncode != 0:
        raise RunnerError(
            f"Command exited with code {returncode}; see {log_path}: {shlex.join(command)}"
        )


def _trajectory_info(batch_dir: Path, instance_id: str) -> tuple[dict[str, Any] | None, str | None]:
    trajectory = batch_dir / instance_id / f"{instance_id}.traj.json"
    if not trajectory.exists():
        return None, None
    payload = _read_json(trajectory)
    if not isinstance(payload, dict) or not isinstance(payload.get("info"), dict):
        raise RunnerError(f"Trajectory has no info object: {trajectory}")
    return payload["info"], str(trajectory)


def validate_predictions(
    predictions: Any,
    expected_ids: Sequence[str],
    *,
    source: Path,
) -> dict[str, dict[str, Any]]:
    if not isinstance(predictions, dict):
        raise RunnerError(f"Predictions must be a JSON object: {source}")
    expected = set(expected_ids)
    actual = set(predictions)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise RunnerError(
            f"Prediction ids do not match {source}: missing={missing[:10]} extra={extra[:10]}"
        )
    for instance_id, prediction in predictions.items():
        if not isinstance(prediction, dict):
            raise RunnerError(f"Prediction for {instance_id} is not a JSON object")
        if prediction.get("instance_id") != instance_id:
            raise RunnerError(f"Prediction instance_id mismatch for {instance_id}")
        if not isinstance(prediction.get("model_patch"), str):
            raise RunnerError(f"Prediction model_patch is not a string for {instance_id}")
        if not isinstance(prediction.get("model_name_or_path"), str):
            raise RunnerError(f"Prediction model_name_or_path is not a string for {instance_id}")
    return predictions


def validate_completed_batch(
    batch_dir: Path, expected_ids: Sequence[str]
) -> dict[str, dict[str, Any]]:
    predictions_path = batch_dir / "preds.json"
    marker_path = batch_dir / "batch_done.json"
    predictions = validate_predictions(
        _read_json(predictions_path), expected_ids, source=predictions_path
    )
    marker = _read_json(marker_path)
    if not isinstance(marker, dict):
        raise RunnerError(f"Completed batch marker must be a JSON object: {marker_path}")
    expected_hash = _canonical_hash(predictions)
    if marker.get("instances") != len(expected_ids) or marker.get(
        "predictions_sha256"
    ) != expected_hash:
        raise RunnerError(
            f"Completed batch marker does not match predictions: {marker_path}"
        )
    return predictions


def merge_batch_predictions(
    run_dir: Path,
    batches: Sequence[Sequence[str]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    combined: dict[str, dict[str, Any]] = {}
    generation_rows: list[dict[str, Any]] = []
    for index, expected_ids in enumerate(batches):
        batch_name = f"batch_{index:03d}"
        batch_dir = run_dir / "batches" / batch_name
        predictions_path = batch_dir / "preds.json"
        predictions = validate_predictions(
            _read_json(predictions_path), expected_ids, source=predictions_path
        )
        overlap = sorted(set(combined) & set(predictions))
        if overlap:
            raise RunnerError(f"Duplicate predictions across batches: {overlap[:10]}")
        combined.update(predictions)

        for instance_id in expected_ids:
            prediction = predictions[instance_id]
            info, trajectory_path = _trajectory_info(batch_dir, instance_id)
            model_stats = (info or {}).get("model_stats") or {}
            if not isinstance(model_stats, dict):
                raise RunnerError(f"model_stats is not an object for {instance_id}")
            if trajectory_path is None:
                status = "parse_failed"
            elif not prediction["model_patch"]:
                status = "model_failed"
            else:
                status = "success"
            generation_rows.append(
                {
                    "instance_id": instance_id,
                    "status": status,
                    "exit_status": (info or {}).get("exit_status"),
                    "has_patch": bool(prediction["model_patch"]),
                    "model_patch_len": len(prediction["model_patch"]),
                    "model_stats": model_stats,
                    "trajectory_path": trajectory_path,
                }
            )
    return combined, generation_rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise RunnerError(f"Required JSONL file does not exist: {path}") from exc
    rows = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RunnerError(f"Invalid JSONL in {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise RunnerError(f"JSONL row is not an object: {path}:{line_number}")
        rows.append(row)
    return rows


def normalize_results(
    *,
    expected_ids: Sequence[str],
    predictions: dict[str, dict[str, Any]],
    generation_rows: Sequence[dict[str, Any]],
    official_report: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    generation_by_id = {row["instance_id"]: row for row in generation_rows}
    if set(generation_by_id) != set(expected_ids):
        raise RunnerError("Generation rows do not exactly cover selected instances")

    report_total = official_report.get("total_instances")
    if report_total != len(expected_ids):
        raise RunnerError(
            f"Official report total_instances={report_total!r}, expected {len(expected_ids)}"
        )
    if official_report.get("submitted_instances") != len(expected_ids):
        raise RunnerError(
            "Official report submitted_instances does not match selected predictions"
        )
    completed = set(official_report.get("completed_ids", []))
    resolved = set(official_report.get("resolved_ids", []))
    unresolved = set(official_report.get("unresolved_ids", []))
    empty_patch = set(official_report.get("empty_patch_ids", []))
    errors = set(official_report.get("error_ids", []))
    if resolved & unresolved:
        raise RunnerError("Official report marks instances as both resolved and unresolved")
    if resolved | unresolved != completed:
        raise RunnerError(
            "Official report completed_ids do not match resolved_ids plus unresolved_ids"
        )
    if (completed & empty_patch) or (completed & errors) or (empty_patch & errors):
        raise RunnerError("Official report outcome id sets overlap")
    count_fields = {
        "completed_instances": len(completed),
        "resolved_instances": len(resolved),
        "unresolved_instances": len(unresolved),
        "empty_patch_instances": len(empty_patch),
        "error_instances": len(errors),
    }
    for field, expected_count in count_fields.items():
        if official_report.get(field) != expected_count:
            raise RunnerError(
                f"Official report {field}={official_report.get(field)!r}, expected {expected_count}"
            )
    known = completed | empty_patch | errors
    unknown_report_ids = known - set(expected_ids)
    if unknown_report_ids:
        raise RunnerError(
            f"Official report contains unknown ids: {sorted(unknown_report_ids)[:10]}"
        )

    rows = []
    for instance_id in expected_ids:
        generation = generation_by_id[instance_id]
        generation_status = generation.get("status")
        if generation_status not in FINAL_STATUSES:
            raise RunnerError(
                f"Generation result has invalid status for {instance_id}: {generation_status!r}"
            )
        if generation_status != "success":
            status = generation_status
        elif instance_id in errors:
            status = "metric_failed"
        elif instance_id in completed:
            status = "success"
        elif instance_id in empty_patch or not generation["has_patch"]:
            status = "model_failed"
        else:
            status = "metric_failed"
        if status not in FINAL_STATUSES:
            raise AssertionError(f"Unknown final status: {status}")
        rows.append(
            {
                "instance_id": instance_id,
                "status": status,
                "resolved": instance_id in resolved if instance_id in completed else None,
                "official_outcome": (
                    "resolved"
                    if instance_id in resolved
                    else "unresolved"
                    if instance_id in unresolved
                    else "empty_patch"
                    if instance_id in empty_patch
                    else "error"
                    if instance_id in errors
                    else "missing"
                ),
                "generation_exit_status": generation["exit_status"],
                "has_patch": generation["has_patch"],
                "model_patch_len": generation["model_patch_len"],
                "model_stats": generation["model_stats"],
                "trajectory_path": generation["trajectory_path"],
                "prediction_model": predictions[instance_id]["model_name_or_path"],
            }
        )

    total_cost = sum(
        float((row.get("model_stats") or {}).get("instance_cost") or 0.0)
        for row in generation_rows
    )
    total_calls = sum(
        int((row.get("model_stats") or {}).get("api_calls") or 0)
        for row in generation_rows
    )
    summary = {
        "total_instances": len(expected_ids),
        "resolved_instances": len(resolved),
        "score": len(resolved) / len(expected_ids),
        "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
        "generation_exit_status_counts": dict(
            sorted(
                Counter(row.get("exit_status") or "missing" for row in generation_rows).items()
            )
        ),
        "total_instance_cost": total_cost,
        "total_api_calls": total_calls,
        "official_report": official_report,
    }
    return rows, summary


class SweBenchLiteRunner:
    def __init__(self, args: argparse.Namespace):
        if not hasattr(args, "offline_dataset"):
            args.offline_dataset = not bool(
                getattr(args, "allow_dataset_download", False)
            )
        if getattr(args, "chain_cache", None) is None:
            args.chain_cache = _env_truthy("SPARSEENGINE_CHAIN_CACHE")
        if args.preserve_thinking is None:
            args.preserve_thinking = bool(args.chain_cache)
        if args.chain_cache and not args.preserve_thinking:
            raise RunnerError(
                "--chain-cache requires --preserve-thinking for append-only prompts; "
                "--no-preserve-thinking is incompatible."
            )
        self.args = args
        self.repo_root = Path(__file__).resolve().parents[2]
        self.swe_bench_dir = (
            args.swe_bench_dir.expanduser().resolve()
            if args.swe_bench_dir is not None
            else None
        )
        self.run_dir = args.run_dir.expanduser().resolve()
        self.status_path = self.run_dir / "status.jsonl"
        self.run_config_path = self.run_dir / "run_config.json"
        self.manifest_path = self.run_dir / "run_manifest.json"
        self.invocations_path = self.run_dir / "invocations.jsonl"
        self.mini_config_path = self.run_dir / "mini_swe_agent_config.yaml"
        self.instances_path = self.run_dir / "instances.txt"
        self.images_path = self.run_dir / "images.txt"
        self.predictions_path = self.run_dir / "preds_all.json"
        self.generation_results_path = self.run_dir / "generation_results.jsonl"
        self.per_sample_results_path = self.run_dir / "per_sample_results.jsonl"
        self.evaluation_identity_path = self.run_dir / "evaluation_identity.json"
        self.official_dir = self.run_dir / "official"
        self.extra_mini_configs = [path.expanduser().resolve() for path in args.mini_extra_config]
        self.extra_mini_config_records: list[dict[str, str]] = []
        self.extra_mini_config_snapshots: list[Path] = []
        self.run_id = args.run_id or f"swe_bench_lite_{self.run_dir.name}"
        self.official_run_id: str | None = None
        if re.fullmatch(r"[A-Za-z0-9_.-]+", self.run_id) is None:
            raise RunnerError(
                "--run-id may only contain letters, numbers, dot, underscore, and dash"
            )
        if args.step_limit <= 0:
            raise RunnerError("--step-limit must be positive")
        if args.wall_time_limit_seconds < 0:
            raise RunnerError(
                "--wall-time-limit-seconds must be non-negative; use 0 to disable it"
            )
        if args.max_tokens <= 0 or args.eval_timeout <= 0 or args.health_timeout <= 0:
            raise RunnerError(
                "token, evaluation-timeout, and health-timeout limits must be positive"
            )
        if args.batch_size <= 0 or args.mini_workers <= 0 or args.eval_workers <= 0:
            raise RunnerError("batch size and worker counts must be positive")
        if args.cost_limit < 0:
            raise RunnerError("--cost-limit must be non-negative")
        if not 0.0 <= args.temperature:
            raise RunnerError("--temperature must be non-negative")
        if not 0.0 < args.top_p <= 1.0:
            raise RunnerError("--top-p must be in (0, 1]")
        if (
            not math.isfinite(args.docker_writable_layer_limit_gib)
            or args.docker_writable_layer_limit_gib < 0
        ):
            raise RunnerError(
                "--docker-writable-layer-limit-gib must be finite and non-negative"
            )
        if (
            not math.isfinite(args.docker_writable_layer_poll_seconds)
            or args.docker_writable_layer_poll_seconds <= 0
        ):
            raise RunnerError(
                "--docker-writable-layer-poll-seconds must be finite and positive"
            )
        if args.cost_tracking == "ignore_errors" and args.cost_limit > 0:
            raise RunnerError(
                "A positive --cost-limit is not reliable with --cost-tracking=ignore_errors; "
                "set --cost-limit=0 or provide model cost metadata and use default tracking"
            )
        prune_policy = getattr(args, "prefix_prune_policy", None)
        if prune_policy:
            if bool(args.chain_cache):
                raise RunnerError("--prefix-prune-policy requires --no-chain-cache")
            range_start = int(args.prefix_prune_range_start)
            range_end = int(args.prefix_prune_range_end)
            keep_tokens = int(args.prefix_prune_keep_tokens)
            trigger_tokens = int(args.prefix_prune_trigger_tokens)
            if (
                range_start < 0
                or range_end <= range_start
                or range_start % 16
                or range_end % 16
            ):
                raise RunnerError("prefix-prune range must be non-empty and 16-token aligned")
            if not 0 <= keep_tokens < range_end - range_start:
                raise RunnerError("prefix-prune keep tokens must be in [0, R-L)")
            if trigger_tokens < range_end:
                raise RunnerError("prefix-prune trigger must be at least range end")
        _reject_secrets(
            {"api_base": args.api_base, "mini_command": args.mini_command},
            source=Path("<command-line arguments>"),
        )

        self.requires_model_api = args.stage in {"prepare", "generate", "all"}
        if args.stage != "summarize" and self.swe_bench_dir is None:
            raise RunnerError("--swe-bench-dir is required unless --stage=summarize")
        if args.stage != "summarize" and not args.model:
            raise RunnerError("--model is required unless --stage=summarize")
        self.api_key = os.environ.get(args.api_key_env, "")
        if self.requires_model_api and not self.api_key:
            raise RunnerError(f"Required API key environment variable is empty: {args.api_key_env}")

        self.rows: list[dict[str, Any]] = []
        self.instance_ids: list[str] = []
        self.image_names: list[str] = []

    def _runtime_provenance(self) -> dict[str, Any]:
        if self.swe_bench_dir is None:
            raise RunnerError("SWE-bench checkout is unavailable for provenance validation")
        return {
            "adapter_git": _git_state(
                self.repo_root,
                source_roots=("benchmark", "scripts/benchmarks"),
            ),
            "swe_bench_git": _git_state(
                self.swe_bench_dir,
                source_roots=("swebench",),
            ),
            "python": {"executable": sys.executable, "version": sys.version},
            "packages": {
                name: _package_version(name)
                for name in ("mini-swe-agent", "swebench", "litellm", "datasets")
            },
        }

    def _validate_existing_provenance(
        self, runtime_provenance: dict[str, Any]
    ) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        manifest = _read_json(self.manifest_path)
        if not isinstance(manifest, dict) or not isinstance(
            manifest.get("runtime_provenance"), dict
        ):
            raise RunnerError(
                f"Existing run manifest lacks runtime provenance: {self.manifest_path}; "
                "use a new --run-dir"
            )
        assert_runtime_provenance_matches(
            manifest["runtime_provenance"], runtime_provenance
        )
        return manifest

    def _status(self, stage: str, status: str, detail: str) -> None:
        _append_jsonl(
            self.status_path,
            {"time": _now(), "stage": stage, "status": status, "detail": detail},
        )

    def _model_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.pop("MINISWE_RECORD_AGENT_TRACE", None)
        if getattr(self.args, "record_agent_trace", False):
            env["MINISWE_RECORD_AGENT_TRACE"] = "1"
        python_path = []
        writable_limit_gib = getattr(
            self.args, "docker_writable_layer_limit_gib", 0.0
        )
        if writable_limit_gib > 0:
            python_path.append(
                str(
                    self.repo_root
                    / "benchmark"
                    / "swe_bench_lite"
                    / "docker_guard_bootstrap"
                )
            )
        python_path.append(str(self.repo_root))
        if env.get("PYTHONPATH"):
            python_path.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(python_path)
        if not self.args.api_proxy_from_environment:
            for key in PROXY_ENV_VARS:
                env.pop(key, None)
        no_proxy = ",".join(
            value for value in (env.get("NO_PROXY"), env.get("no_proxy")) if value
        )
        no_proxy_values = [item for item in no_proxy.split(",") if item]
        for host in sorted(LOCAL_HOSTS):
            if host not in no_proxy_values:
                no_proxy_values.append(host)
        env["NO_PROXY"] = ",".join(no_proxy_values)
        env["no_proxy"] = env["NO_PROXY"]
        if self.args.offline_dataset:
            env["HF_DATASETS_OFFLINE"] = "1"
            env["HF_HUB_OFFLINE"] = "1"
        else:
            env.pop("HF_DATASETS_OFFLINE", None)
            env.pop("HF_HUB_OFFLINE", None)
        env.pop("MSWEA_GLOBAL_CALL_LIMIT", None)
        env.pop("MSWEA_GLOBAL_COST_LIMIT", None)
        if bool(getattr(self.args, "chain_cache", False)):
            env["SPARSEENGINE_CHAIN_CACHE"] = "1"
        else:
            env.pop("SPARSEENGINE_CHAIN_CACHE", None)
        prune_env_vars = (
            "SPARSEENGINE_PREFIX_PRUNE_POLICY",
            "SPARSEENGINE_PREFIX_PRUNE_TRIGGER_TOKENS",
            "SPARSEENGINE_PREFIX_PRUNE_RANGE_START",
            "SPARSEENGINE_PREFIX_PRUNE_RANGE_END",
            "SPARSEENGINE_PREFIX_PRUNE_KEEP_TOKENS",
            "SPARSEENGINE_PREFIX_PRUNE_EVENTS",
        )
        prune_policy = getattr(self.args, "prefix_prune_policy", None)
        if prune_policy:
            env["SPARSEENGINE_PREFIX_PRUNE_POLICY"] = str(prune_policy)
            env["SPARSEENGINE_PREFIX_PRUNE_TRIGGER_TOKENS"] = str(
                self.args.prefix_prune_trigger_tokens
            )
            env["SPARSEENGINE_PREFIX_PRUNE_RANGE_START"] = str(
                self.args.prefix_prune_range_start
            )
            env["SPARSEENGINE_PREFIX_PRUNE_RANGE_END"] = str(
                self.args.prefix_prune_range_end
            )
            env["SPARSEENGINE_PREFIX_PRUNE_KEEP_TOKENS"] = str(
                self.args.prefix_prune_keep_tokens
            )
            env["SPARSEENGINE_PREFIX_PRUNE_EVENTS"] = str(
                self.run_dir / "prefix_prune_events.jsonl"
            )
        else:
            for key in prune_env_vars:
                env.pop(key, None)
        guard_env_vars = (
            "SPARSEENGINE_DOCKER_WRITABLE_LAYER_LIMIT_BYTES",
            "SPARSEENGINE_DOCKER_WRITABLE_LAYER_POLL_SECONDS",
            "SPARSEENGINE_DOCKER_WRITABLE_LAYER_EVENTS",
            "SPARSEENGINE_SWE_RUN_ID",
        )
        if writable_limit_gib > 0:
            env["SPARSEENGINE_DOCKER_WRITABLE_LAYER_LIMIT_BYTES"] = str(
                int(writable_limit_gib * 1024**3)
            )
            env["SPARSEENGINE_DOCKER_WRITABLE_LAYER_POLL_SECONDS"] = str(
                getattr(self.args, "docker_writable_layer_poll_seconds", 1.0)
            )
            env["SPARSEENGINE_DOCKER_WRITABLE_LAYER_EVENTS"] = str(
                self.run_dir / "docker_writable_layer_guard.jsonl"
            )
            env["SPARSEENGINE_SWE_RUN_ID"] = self.run_id
        else:
            for key in guard_env_vars:
                env.pop(key, None)
        return env

    def _load_selection(self) -> None:
        if self.swe_bench_dir is None or not self.swe_bench_dir.is_dir():
            raise RunnerError(f"SWE-bench checkout does not exist: {self.swe_bench_dir}")
        if self.args.offline_dataset:
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            os.environ.pop("HF_DATASETS_OFFLINE", None)
            os.environ.pop("HF_HUB_OFFLINE", None)
        rows = _load_dataset(self.args.dataset, self.args.split)
        self.rows = select_rows(
            rows,
            instance_ids_file=self.args.instance_ids_file,
            instance_filter=self.args.filter,
            slice_spec=self.args.slice,
        )
        self.instance_ids = [row["instance_id"] for row in self.rows]
        self.image_names = _instance_image_names(self.rows)

    def _prepare_extra_mini_configs(self) -> None:
        records = []
        snapshots = []
        snapshot_dir = self.run_dir / "mini_extra_configs"
        for index, source in enumerate(self.extra_mini_configs):
            try:
                content = source.read_text(encoding="utf-8")
            except FileNotFoundError as exc:
                raise RunnerError(f"Extra mini-SWE-agent config does not exist: {source}") from exc
            try:
                parsed = yaml.safe_load(content)
            except yaml.YAMLError as exc:
                raise RunnerError(
                    f"Invalid YAML in extra mini-SWE-agent config {source}: {exc}"
                ) from exc
            _reject_secrets(content, source=source)
            _reject_secrets(parsed, source=source)
            snapshot = snapshot_dir / f"{index:02d}_{source.name}"
            if snapshot.exists() and snapshot.read_text(encoding="utf-8") != content:
                raise RunnerError(f"Extra mini-SWE-agent config snapshot changed: {snapshot}")
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_text(content, encoding="utf-8")
            records.append(
                {
                    "source": str(source),
                    "snapshot": str(snapshot),
                    "sha256": _sha256_bytes(content.encode("utf-8")),
                }
            )
            snapshots.append(snapshot)
        self.extra_mini_config_records = records
        self.extra_mini_config_snapshots = snapshots

    def _semantic_config(self, server_manifest: dict[str, Any] | None) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dataset": self.args.dataset,
            "split": self.args.split,
            "instance_ids": self.instance_ids,
            "dataset_rows_sha256": _canonical_hash(self.rows),
            "model": self.args.model,
            "api_base": self.args.api_base,
            "served_model_name": self.args.served_model_name,
            "mini_base_config": self.args.mini_base_config,
            "mini_extra_configs": self.extra_mini_config_records,
            "mini_command": self.args.mini_command,
            "record_agent_trace": bool(getattr(self.args, "record_agent_trace", False)),
            "batch_size": self.args.batch_size,
            "step_limit": self.args.step_limit,
            "cost_limit": self.args.cost_limit,
            "wall_time_limit_seconds": self.args.wall_time_limit_seconds,
            "cost_tracking": self.args.cost_tracking,
            "max_tokens": self.args.max_tokens,
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "enable_thinking": self.args.enable_thinking,
            "preserve_thinking": self.args.preserve_thinking,
            "chain_cache": bool(self.args.chain_cache),
            "prefix_prune": (
                {
                    "policy": self.args.prefix_prune_policy,
                    "trigger_tokens": self.args.prefix_prune_trigger_tokens,
                    "range": [
                        self.args.prefix_prune_range_start,
                        self.args.prefix_prune_range_end,
                    ],
                    "keep_tokens": self.args.prefix_prune_keep_tokens,
                }
                if getattr(self.args, "prefix_prune_policy", None)
                else None
            ),
            "docker_writable_layer_limit_gib": self.args.docker_writable_layer_limit_gib,
            "docker_writable_layer_poll_seconds": self.args.docker_writable_layer_poll_seconds,
            "seed": None,
            "seed_control": "not exposed by this adapter",
            "eval_timeout": self.args.eval_timeout,
            "server_manifest_sha256": _canonical_hash(server_manifest) if server_manifest else None,
        }

    def prepare(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._status("prepare", "running", "validating dataset, server, and Docker images")
        runtime_provenance = self._runtime_provenance()
        self._validate_existing_provenance(runtime_provenance)
        self._load_selection()
        self._prepare_extra_mini_configs()
        self.instances_path.write_text("\n".join(self.instance_ids) + "\n", encoding="utf-8")
        self.images_path.write_text("\n".join(self.image_names) + "\n", encoding="utf-8")

        server_manifest = None
        parsed_api = urllib.parse.urlparse(self.args.api_base) if self.args.api_base else None
        is_local_api = parsed_api is not None and parsed_api.hostname in LOCAL_HOSTS
        if is_local_api and self.args.server_manifest is None:
            raise RunnerError("A local --api-base requires --server-manifest")
        if self.args.server_manifest is not None:
            if not self.args.api_base:
                raise RunnerError("--server-manifest requires --api-base")
            server_manifest = _validate_local_server_manifest(
                self.args.server_manifest,
                api_base=self.args.api_base,
                served_model_name=self.args.served_model_name,
            )

        if self.args.api_base and self.requires_model_api:
            check_openai_server(
                self.args.api_base,
                api_key=self.api_key,
                served_model_name=self.args.served_model_name,
                timeout=self.args.health_timeout,
                use_environment_proxy=self.args.api_proxy_from_environment,
            )
        requires_images = self.args.stage in {"prepare", "generate", "evaluate", "all"}
        if requires_images and not self.args.allow_image_pulls:
            _require_local_images(self.image_names)

        semantic_config = self._semantic_config(server_manifest)
        if self.run_config_path.exists():
            existing = _read_json(self.run_config_path)
            if existing != semantic_config:
                raise RunnerError(
                    f"Run configuration differs from existing {self.run_config_path}; "
                    "use a new --run-dir instead of mixing experiments"
                )
        else:
            _write_json(self.run_config_path, semantic_config)

        config_text = render_mini_config(
            step_limit=self.args.step_limit,
            cost_limit=self.args.cost_limit,
            wall_time_limit_seconds=self.args.wall_time_limit_seconds,
            cost_tracking=self.args.cost_tracking,
            max_tokens=self.args.max_tokens,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            enable_thinking=self.args.enable_thinking,
            preserve_thinking=self.args.preserve_thinking,
            api_base=self.args.api_base,
        )
        if (
            self.mini_config_path.exists()
            and self.mini_config_path.read_text(encoding="utf-8") != config_text
        ):
            raise RunnerError(f"Generated mini-SWE-agent config changed: {self.mini_config_path}")
        self.mini_config_path.write_text(config_text, encoding="utf-8")

        if server_manifest is not None:
            _write_json(self.run_dir / "server_manifest.json", server_manifest)
        if not self.manifest_path.exists():
            _write_json(
                self.manifest_path,
                {
                    "created_at": _now(),
                    "runtime_provenance": runtime_provenance,
                    "run_config": semantic_config,
                    "run_id": self.run_id,
                    "run_dir": str(self.run_dir),
                    "api_key_env": self.args.api_key_env,
                    "api_proxy_from_environment": self.args.api_proxy_from_environment,
                    "offline_dataset": self.args.offline_dataset,
                },
            )
        _append_jsonl(
            self.invocations_path,
            {
                "time": _now(),
                "argv": sys.argv,
                "mini_workers": self.args.mini_workers,
                "eval_workers": self.args.eval_workers,
                "batch_size": self.args.batch_size,
                "stage": self.args.stage,
            },
        )
        self._status(
            "prepare",
            "completed",
            f"instances={len(self.instance_ids)} "
            f"images_checked={requires_images and not self.args.allow_image_pulls}",
        )

    def _batches(self) -> list[list[str]]:
        return [
            self.instance_ids[index : index + self.args.batch_size]
            for index in range(0, len(self.instance_ids), self.args.batch_size)
        ]

    def _load_artifact_context(self) -> None:
        config = _read_json(self.run_config_path)
        if not isinstance(config, dict):
            raise RunnerError(f"Run configuration must be a JSON object: {self.run_config_path}")
        required = {
            "run_id",
            "instance_ids",
            "batch_size",
            "model",
            "dataset",
            "split",
        }
        missing = sorted(required - set(config))
        if missing:
            raise RunnerError(f"Run configuration is missing fields: {missing}")
        if self.args.run_id is not None and self.args.run_id != config["run_id"]:
            raise RunnerError(
                f"--run-id {self.args.run_id!r} does not match stored run id "
                f"{config['run_id']!r}"
            )
        if not isinstance(config["run_id"], str) or re.fullmatch(
            r"[A-Za-z0-9_.-]+", config["run_id"]
        ) is None:
            raise RunnerError("Stored run_id is invalid")
        if (
            not isinstance(config["instance_ids"], list)
            or not config["instance_ids"]
            or not all(
                isinstance(instance_id, str) and instance_id
                for instance_id in config["instance_ids"]
            )
            or len(set(config["instance_ids"])) != len(config["instance_ids"])
        ):
            raise RunnerError("Stored instance_ids must be unique non-empty strings")
        if (
            not isinstance(config["batch_size"], int)
            or isinstance(config["batch_size"], bool)
            or config["batch_size"] <= 0
        ):
            raise RunnerError("Stored batch_size must be a positive integer")
        for field in ("model", "dataset", "split"):
            if not isinstance(config[field], str) or not config[field]:
                raise RunnerError(f"Stored {field} must be a non-empty string")
        self.run_id = config["run_id"]
        self.instance_ids = config["instance_ids"]
        self.args.batch_size = config["batch_size"]
        self.args.model = config["model"]
        self.args.dataset = config["dataset"]
        self.args.split = config["split"]

        identity = _read_json(self.evaluation_identity_path)
        if (
            not isinstance(identity, dict)
            or not isinstance(identity.get("official_run_id"), str)
            or not isinstance(identity.get("predictions_sha256"), str)
        ):
            raise RunnerError(
                f"Evaluation identity is missing or invalid: {self.evaluation_identity_path}"
            )
        predictions = validate_predictions(
            _read_json(self.predictions_path),
            self.instance_ids,
            source=self.predictions_path,
        )
        if _canonical_hash(predictions) != identity["predictions_sha256"]:
            raise RunnerError("Predictions do not match the stored evaluation identity")
        expected_official_run_id = build_official_run_id(
            self.run_id, identity["predictions_sha256"]
        )
        if identity["official_run_id"] != expected_official_run_id:
            raise RunnerError("Stored official run id does not match the prediction hash")
        self.official_run_id = identity["official_run_id"]

    def _prepare_evaluation_identity(
        self, predictions: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        merge_summary = _read_json(self.run_dir / "prediction_merge_summary.json")
        if not isinstance(merge_summary, dict):
            raise RunnerError("Prediction merge summary must be a JSON object")
        predictions_sha256 = _canonical_hash(predictions)
        if predictions_sha256 != merge_summary.get("predictions_sha256"):
            raise RunnerError("preds_all.json changed after generation merge")

        manifest = _read_json(self.manifest_path)
        if not isinstance(manifest, dict) or not isinstance(
            manifest.get("runtime_provenance"), dict
        ):
            raise RunnerError(f"Run manifest lacks runtime provenance: {self.manifest_path}")
        official_run_id = build_official_run_id(self.run_id, predictions_sha256)
        identity = {
            "logical_run_id": self.run_id,
            "official_run_id": official_run_id,
            "predictions_sha256": predictions_sha256,
            "runtime_provenance_sha256": _canonical_hash(manifest["runtime_provenance"]),
        }
        if self.evaluation_identity_path.exists():
            existing = _read_json(self.evaluation_identity_path)
            if existing != identity:
                raise RunnerError(
                    f"Evaluation identity differs from {self.evaluation_identity_path}; "
                    "use a new --run-dir"
                )
        else:
            _write_json(self.evaluation_identity_path, identity)

        log_root = self.official_dir / "logs" / "run_evaluation" / official_run_id
        marker_path = log_root / ".sparseengine_adapter_identity.json"
        if log_root.exists() and not marker_path.exists() and any(log_root.iterdir()):
            raise RunnerError(
                f"Refusing unowned SWE-bench cache directory without identity marker: {log_root}"
            )
        if marker_path.exists():
            marker = _read_json(marker_path)
            if marker != identity:
                raise RunnerError(f"SWE-bench cache identity mismatch: {marker_path}")
        else:
            _write_json(marker_path, identity)

        self.official_run_id = official_run_id
        return identity

    def _write_generation_failure_results(
        self, *, failed_batch_index: int, error: Exception
    ) -> None:
        rows = []
        for batch_index, instance_ids in enumerate(self._batches()):
            batch_dir = self.run_dir / "batches" / f"batch_{batch_index:03d}"
            predictions = {}
            predictions_path = batch_dir / "preds.json"
            if predictions_path.exists():
                try:
                    loaded = _read_json(predictions_path)
                    if isinstance(loaded, dict):
                        predictions = loaded
                except RunnerError:
                    predictions = {}
            for instance_id in instance_ids:
                prediction = predictions.get(instance_id)
                try:
                    info, trajectory_path = _trajectory_info(batch_dir, instance_id)
                except RunnerError:
                    info, trajectory_path = None, None
                patch = prediction.get("model_patch") if isinstance(prediction, dict) else None
                if failed_batch_index < 0:
                    status = "invalid_input"
                    detail = str(error)
                elif batch_index > failed_batch_index:
                    status = "skipped_by_policy"
                    detail = "not attempted after an earlier generation batch failed"
                elif trajectory_path is None:
                    status = "parse_failed" if prediction is not None else "model_failed"
                    detail = str(error)
                elif not isinstance(patch, str) or not patch:
                    status = "model_failed"
                    detail = str(error)
                else:
                    status = "success"
                    detail = None
                rows.append(
                    {
                        "instance_id": instance_id,
                        "status": status,
                        "exit_status": (info or {}).get("exit_status"),
                        "has_patch": isinstance(patch, str) and bool(patch),
                        "model_patch_len": len(patch) if isinstance(patch, str) else 0,
                        "model_stats": (info or {}).get("model_stats") or {},
                        "trajectory_path": trajectory_path,
                        "error": detail,
                    }
                )
        _write_jsonl(self.generation_results_path, rows)

    def _write_evaluation_failure_results(self, error: Exception) -> None:
        generation_rows = _read_jsonl(self.generation_results_path)
        generation_by_id = {row.get("instance_id"): row for row in generation_rows}
        if set(generation_by_id) != set(self.instance_ids):
            raise RunnerError(
                "Cannot write evaluation failure statuses because generation results "
                "do not cover the selected instances"
            ) from error
        rows = []
        for instance_id in self.instance_ids:
            generation = generation_by_id[instance_id]
            generation_status = generation.get("status")
            status = (
                "metric_failed" if generation_status == "success" else generation_status
            )
            if status not in FINAL_STATUSES:
                status = "parse_failed"
            rows.append(
                {
                    "instance_id": instance_id,
                    "status": status,
                    "resolved": None,
                    "official_outcome": "error",
                    "generation_exit_status": generation.get("exit_status"),
                    "has_patch": generation.get("has_patch", False),
                    "model_patch_len": generation.get("model_patch_len", 0),
                    "model_stats": generation.get("model_stats") or {},
                    "trajectory_path": generation.get("trajectory_path"),
                    "error": str(error),
                }
            )
        _write_jsonl(self.per_sample_results_path, rows)

    def generate(self) -> None:
        batches = self._batches()
        model_env = self._model_env()
        mini_prefix = shlex.split(self.args.mini_command)
        if getattr(self.args, "record_agent_trace", False):
            timed = self.repo_root / "scripts/official_experiments/chain_cache_miniswe/timed_mini.py"
            if mini_prefix != ["mini-extra"] and mini_prefix != [sys.executable, str(timed)]:
                raise RunnerError("--record-agent-trace requires the default mini-extra or timed_mini command")
            mini_prefix = [sys.executable, str(timed)]
        if not mini_prefix:
            error = RunnerError("--mini-command is empty")
            self._write_generation_failure_results(
                failed_batch_index=-1, error=error
            )
            raise error

        for index, instance_ids in enumerate(batches):
            batch_name = f"batch_{index:03d}"
            batch_dir = self.run_dir / "batches" / batch_name
            batch_dir.mkdir(parents=True, exist_ok=True)
            ids_path = batch_dir / "instances.txt"
            ids_path.write_text("\n".join(instance_ids) + "\n", encoding="utf-8")
            done_path = batch_dir / "batch_done.json"
            predictions_path = batch_dir / "preds.json"
            if done_path.exists():
                try:
                    validate_completed_batch(batch_dir, instance_ids)
                except Exception as exc:
                    self._write_generation_failure_results(
                        failed_batch_index=index, error=exc
                    )
                    self._status(batch_name, "failed", str(exc))
                    raise
                self._status(batch_name, "skipped", "validated completed batch")
                continue

            instance_regex = "^(?:" + "|".join(re.escape(item) for item in instance_ids) + ")$"
            command = [
                *mini_prefix,
                "swebench",
                "--subset",
                self.args.dataset,
                "--split",
                self.args.split,
                "--filter",
                instance_regex,
                "--workers",
                str(self.args.mini_workers),
                "--output",
                str(batch_dir),
                "--model",
                self.args.model,
                "--environment-class",
                "docker",
                "-c",
                self.args.mini_base_config,
            ]
            for extra_config in self.extra_mini_config_snapshots:
                command.extend(["-c", str(extra_config)])
            command.extend(["-c", str(self.mini_config_path)])
            self._status(
                batch_name,
                "running",
                f"instances={len(instance_ids)} workers={self.args.mini_workers}",
            )
            try:
                _run_logged(
                    command,
                    cwd=self.swe_bench_dir,
                    env=model_env,
                    log_path=self.run_dir / "logs" / f"{batch_name}_mini.log",
                )
                predictions = validate_predictions(
                    _read_json(predictions_path), instance_ids, source=predictions_path
                )
                _write_json(
                    done_path,
                    {
                        "completed_at": _now(),
                        "instances": len(instance_ids),
                        "predictions_sha256": _canonical_hash(predictions),
                    },
                )
            except Exception as exc:
                self._write_generation_failure_results(
                    failed_batch_index=index, error=exc
                )
                self._status(batch_name, "failed", str(exc))
                raise
            self._status(batch_name, "completed", f"instances={len(instance_ids)}")

        self._status("merge", "running", "validating and merging batch predictions")
        try:
            combined, generation_rows = merge_batch_predictions(self.run_dir, batches)
            validate_predictions(combined, self.instance_ids, source=self.predictions_path)
            _write_json(self.predictions_path, combined)
            _write_jsonl(self.generation_results_path, generation_rows)
            _write_json(
                self.run_dir / "prediction_merge_summary.json",
                {
                    "expected": len(self.instance_ids),
                    "combined": len(combined),
                    "predictions_sha256": _canonical_hash(combined),
                    "generation_results_sha256": _sha256_file(self.generation_results_path),
                },
            )
        except Exception as exc:
            self._write_generation_failure_results(
                failed_batch_index=len(batches), error=exc
            )
            self._status("merge", "failed", str(exc))
            raise
        self._status("merge", "completed", f"predictions={len(combined)}")

    def _official_report_path(self) -> Path:
        if self.official_run_id is None:
            raise RunnerError("Official evaluation identity has not been initialized")
        candidates = sorted(self.official_dir.glob(f"*.{self.official_run_id}.json"))
        if len(candidates) != 1:
            raise RunnerError(
                f"Expected one official report matching *.{self.official_run_id}.json in "
                f"{self.official_dir}, found {len(candidates)}"
            )
        return candidates[0]

    def evaluate(self) -> None:
        self._status(
            "evaluate",
            "running",
            f"instances={len(self.instance_ids)} workers={self.args.eval_workers}",
        )
        try:
            predictions = validate_predictions(
                _read_json(self.predictions_path),
                self.instance_ids,
                source=self.predictions_path,
            )
            self.official_dir.mkdir(parents=True, exist_ok=True)
            identity = self._prepare_evaluation_identity(predictions)
            for stale_report in self.official_dir.glob(
                f"*.{self.official_run_id}.json"
            ):
                stale_report.unlink()
            command = [
                sys.executable,
                "-m",
                "swebench.harness.run_evaluation",
                "--dataset_name",
                self.args.dataset,
                "--split",
                self.args.split,
                "--predictions_path",
                str(self.predictions_path),
                "--max_workers",
                str(self.args.eval_workers),
                "--timeout",
                str(self.args.eval_timeout),
                "--run_id",
                identity["official_run_id"],
                "--report_dir",
                str(self.official_dir),
                "--instance_ids",
                *self.instance_ids,
            ]
            _run_logged(
                command,
                cwd=self.official_dir,
                env=self._model_env(),
                log_path=self.run_dir / "logs" / "official_eval.log",
            )
            report_path = self._official_report_path()
            report = _read_json(report_path)
            if not isinstance(report, dict) or report.get("total_instances") != len(
                self.instance_ids
            ):
                raise RunnerError(
                    f"Official report does not cover selected instances: {report_path}"
                )
        except Exception as exc:
            self._status("evaluate", "failed", str(exc))
            raise
        self._status("evaluate", "completed", f"report={report_path}")

    def summarize(self) -> None:
        batches = self._batches()
        predictions = validate_predictions(
            _read_json(self.predictions_path), self.instance_ids, source=self.predictions_path
        )
        _, generation_rows = merge_batch_predictions(self.run_dir, batches)
        report_path = self._official_report_path()
        official_report = _read_json(report_path)
        if not isinstance(official_report, dict):
            raise RunnerError(f"Official report must be a JSON object: {report_path}")
        per_sample, summary = normalize_results(
            expected_ids=self.instance_ids,
            predictions=predictions,
            generation_rows=generation_rows,
            official_report=official_report,
        )
        _write_jsonl(self.per_sample_results_path, per_sample)
        summary.update(
            {
                "written_at": _now(),
                "run_id": self.run_id,
                "official_run_id": self.official_run_id,
                "run_dir": str(self.run_dir),
                "model": self.args.model,
                "dataset": self.args.dataset,
                "split": self.args.split,
                "official_report_path": str(report_path),
                "official_eval_log_path": str(self.run_dir / "logs" / "official_eval.log"),
                "official_instance_log_root": str(
                    self.official_dir
                    / "logs"
                    / "run_evaluation"
                    / str(self.official_run_id)
                ),
                "predictions_path": str(self.predictions_path),
                "generation_results_path": str(self.generation_results_path),
                "per_sample_results_path": str(self.per_sample_results_path),
                "run_config_path": str(self.run_config_path),
                "run_manifest_path": str(self.manifest_path),
            }
        )
        _write_json(self.run_dir / "final_summary.json", summary)
        self._status(
            "summarize",
            "completed",
            f"resolved={summary['resolved_instances']}/{summary['total_instances']}",
        )

    def run(self) -> None:
        if self.args.stage == "summarize":
            try:
                self._load_artifact_context()
                self.summarize()
            except Exception as exc:
                self._status("summarize", "failed", str(exc))
                if self.generation_results_path.exists() and self.instance_ids:
                    self._write_evaluation_failure_results(exc)
                raise
            return
        try:
            self.prepare()
        except Exception as exc:
            self._status("prepare", "failed", str(exc))
            raise
        if self.args.stage in {"generate", "all"}:
            self.generate()
        if self.args.stage in {"evaluate", "all"}:
            try:
                self.evaluate()
            except Exception as exc:
                if self.generation_results_path.exists():
                    self._write_evaluation_failure_results(exc)
                raise
        if self.args.stage in {"summarize", "all"}:
            try:
                self.summarize()
            except Exception as exc:
                self._status("summarize", "failed", str(exc))
                raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run mini-SWE-agent and the official SWE-bench Lite harness without "
            "vendoring either project."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("prepare", "generate", "evaluate", "summarize", "all"),
        default="all",
    )
    parser.add_argument("--swe-bench-dir", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dataset", default="SWE-bench/SWE-bench_Lite")
    parser.add_argument("--split", default="test")
    parser.add_argument("--instance-ids-file", type=Path, default=None)
    parser.add_argument("--filter", default=None, help="Regex applied before --slice.")
    parser.add_argument("--slice", default=None, help="START:STOP applied after --filter.")

    parser.add_argument(
        "--model", default=None, help="LiteLLM model id, e.g. openai/sparseengine-swe."
    )
    parser.add_argument(
        "--api-base", default=None, help="OpenAI-compatible API base ending in /v1."
    )
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--server-manifest", type=Path, default=None)
    parser.add_argument("--health-timeout", type=float, default=10.0)
    parser.add_argument(
        "--api-proxy-from-environment",
        action="store_true",
        help="Keep HTTP proxy variables for model API calls. Direct access is the default.",
    )

    parser.add_argument("--mini-command", default="mini-extra")
    parser.add_argument("--record-agent-trace", action="store_true",
                        help="Record per-instance non-streaming HTTP input/output and inter-turn delays.")
    parser.add_argument("--mini-base-config", default="swebench.yaml")
    parser.add_argument(
        "--mini-extra-config",
        action="append",
        type=Path,
        default=[],
        help="Additional provider config merged before the generated shared config.",
    )
    parser.add_argument("--mini-workers", type=int, default=1)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--step-limit", type=int, default=80)
    parser.add_argument("--cost-limit", type=float, default=0.0)
    parser.add_argument(
        "--wall-time-limit-seconds",
        type=int,
        default=1800,
        help="Per-agent wall-clock limit in seconds; 0 disables the limit.",
    )
    parser.add_argument("--eval-timeout", type=int, default=1800)
    parser.add_argument("--cost-tracking", choices=("default", "ignore_errors"), default=None)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable model thinking through the chat template.",
    )
    parser.add_argument(
        "--preserve-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Keep prior assistant reasoning in multi-turn chat prompts. "
            "Defaults to enabled with chain cache, disabled otherwise; "
            "chain cache rejects --no-preserve-thinking."
        ),
    )
    parser.add_argument(
        "--chain-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable SparseEngine chain reuse in the mini-SWE-agent model adapter. "
            "When omitted, SPARSEENGINE_CHAIN_CACHE is read once for compatibility; "
            "the resolved value is persisted in run_config.json."
        ),
    )
    parser.add_argument(
        "--prefix-prune-policy",
        choices=("snapkv_global", "kvzip_global"),
        default=None,
        help="Prune each MiniSWE radix-tree path once and verify reuse on its next turn.",
    )
    parser.add_argument("--prefix-prune-trigger-tokens", type=int, default=4096)
    parser.add_argument("--prefix-prune-range-start", type=int, default=512)
    parser.add_argument("--prefix-prune-range-end", type=int, default=4096)
    parser.add_argument("--prefix-prune-keep-tokens", type=int, default=1792)
    parser.add_argument(
        "--docker-writable-layer-limit-gib",
        type=float,
        default=4.0,
        help=(
            "Kill a sample container and fail that sample when its Docker writable "
            "layer exceeds this many GiB (default: 4); 0 disables the guard."
        ),
    )
    parser.add_argument(
        "--docker-writable-layer-poll-seconds",
        type=float,
        default=1.0,
        help="Polling interval for the Docker writable-layer guard.",
    )
    parser.add_argument(
        "--allow-image-pulls",
        action="store_true",
        help="Skip the local-image check and allow external tools to pull/build missing images.",
    )
    parser.add_argument(
        "--allow-dataset-download",
        action="store_true",
        help="Allow Hugging Face access instead of forcing cached/offline dataset loading.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.served_model_name is None and args.model is not None:
        args.served_model_name = args.model.rsplit("/", 1)[-1]
    if args.cost_tracking is None:
        args.cost_tracking = "ignore_errors" if args.api_base else "default"
    args.offline_dataset = not args.allow_dataset_download
    try:
        SweBenchLiteRunner(args).run()
    except RunnerError as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
