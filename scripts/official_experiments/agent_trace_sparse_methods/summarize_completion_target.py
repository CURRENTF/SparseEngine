"""Stop a forced agent replay at a fixed number of successful server completions."""

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
import re
import signal
import time


TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
REQUEST_ID = re.compile(r"request_finish id=(\S+)")
PROMPT_TOKENS = re.compile(r"\bprompt_tokens=(\d+)")
COMPLETION_TOKENS = re.compile(r"\bcompletion_tokens=(\d+)")
HTTP_ERROR = re.compile(r'HTTP/1\.1" [45]\d\d\b')
HTTP_SUCCESS = 'POST /v1/chat/completions HTTP/1.1" 200'
PREEMPTION = "驱逐请求 id ="


def client_running(pid: int) -> bool:
    stat = Path(f"/proc/{pid}/stat")
    return stat.exists() and stat.read_text().split(") ", 1)[1][0] != "Z"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--client-pid", type=int, required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--trace-manifest", type=Path, required=True)
    parser.add_argument("--forced-workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.target <= 0:
        parser.error("--target must be positive")

    trace_hash = hashlib.sha256(args.trace_manifest.read_bytes()).hexdigest()
    forced = json.loads(args.forced_workload.read_text())
    if forced["trace_sha256"] != trace_hash:
        raise ValueError("Forced workload and trace manifest differ")

    first_start = None
    last_finish = None
    finished = set()
    prompt_tokens = 0
    output_tokens = 0
    cancels = 0
    failures = 0
    preemptions = 0
    recompute_replays = 0
    http_errors = 0
    http_successes = 0
    offset = 0
    last_progress = time.monotonic()
    started = last_progress
    while len(finished) < args.target or http_successes < args.target:
        with args.server_log.open("r", errors="replace") as log:
            log.seek(offset)
            while line := log.readline():
                before_target = len(finished) < args.target
                if before_target and HTTP_ERROR.search(line):
                    http_errors += 1
                if HTTP_SUCCESS in line:
                    http_successes += 1
                    last_progress = time.monotonic()
                timestamp = TIMESTAMP.search(line)
                if not timestamp:
                    if len(finished) >= args.target and http_successes >= args.target:
                        break
                    continue
                if "request_start id=" in line and first_start is None:
                    first_start = datetime.fromisoformat(timestamp.group().replace(" ", "T"))
                if "request_finish id=" in line and before_target:
                    request_id = REQUEST_ID.search(line).group(1)
                    if request_id in finished:
                        raise ValueError(f"Duplicate finished request: {request_id}")
                    finished.add(request_id)
                    prompt_tokens += int(PROMPT_TOKENS.search(line).group(1))
                    output_tokens += int(COMPLETION_TOKENS.search(line).group(1))
                    last_finish = datetime.fromisoformat(timestamp.group().replace(" ", "T"))
                    last_progress = time.monotonic()
                elif before_target and "request_cancel id=" in line:
                    cancels += 1
                elif before_target and ("request_failure id=" in line or "request_error id=" in line):
                    failures += 1
                elif before_target and PREEMPTION in line:
                    preemptions += 1
                elif before_target and "recompute_replay_start seq_id=" in line:
                    recompute_replays += 1
                if len(finished) >= args.target and http_successes >= args.target:
                    break
            offset = log.tell()
        if len(finished) >= args.target and http_successes >= args.target:
            break
        if not client_running(args.client_pid):
            raise RuntimeError(
                f"Replay client exited at {len(finished)} request_finish records "
                f"and {http_successes} HTTP 200 responses of {args.target}"
            )
        if time.monotonic() - last_progress > 1800:
            raise TimeoutError(f"No completed request for 30 minutes: {len(finished)} of {args.target}")
        if time.monotonic() - started > 43200:
            raise TimeoutError(f"Completion target not reached after 12 hours: {len(finished)} of {args.target}")
        time.sleep(1)

    if first_start is None or last_finish is None:
        raise ValueError("Missing request timing boundary")
    if failures or cancels or http_errors:
        raise ValueError(
            f"Replay errors before target: failures={failures}, cancels={cancels}, "
            f"HTTP errors={http_errors}"
        )
    if preemptions or recompute_replays:
        raise ValueError(
            "Active requests were preempted before target: "
            f"preemptions={preemptions} recompute_replays={recompute_replays}"
        )
    result = {
        "schema": "agent_trace_time_to_completed_requests_v1",
        "status": "target_reached",
        "method": args.method,
        "concurrency": args.concurrency,
        "request_target": args.target,
        "timing_boundary": f"server_log_first_request_start_to_{args.target}th_request_finish_second_resolution",
        "first_request_start": first_start.isoformat(),
        "target_request_finish": last_finish.isoformat(),
        "elapsed_s": (last_finish - first_start).total_seconds(),
        "completed_request_prompt_tokens": prompt_tokens,
        "completed_request_output_tokens": output_tokens,
        "errors_before_target": failures + http_errors,
        "successful_http_responses_at_target": http_successes,
        "request_cancels_before_target": cancels,
        "preemptions_before_target": preemptions,
        "recompute_replays_before_target": recompute_replays,
        "trace_manifest_sha256": trace_hash,
        "forced_workload_sha256": forced["forced_workload_sha256"],
    }
    temp = args.output.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temp, args.output)
    os.kill(args.client_pid, signal.SIGTERM)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
