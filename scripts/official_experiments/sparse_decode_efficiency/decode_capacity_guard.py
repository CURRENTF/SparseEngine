"""Keep a visible reservation between cases and reject external contention."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def record_contention(path, foreign_pids, episodes, active_pids):
    """Atomically retain bounded contention episodes for later validity checks."""
    now = time.time()
    current = tuple(sorted(foreign_pids))
    if current != active_pids:
        episodes.append({
            "foreign_pids": list(current),
            "first_seen_unix_s": now,
            "last_seen_unix_s": now,
            "samples": 1,
        })
    else:
        episodes[-1]["last_seen_unix_s"] = now
        episodes[-1]["samples"] += 1
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({
        "policy": "mark_potentially_invalid_and_continue",
        "episodes": episodes,
    }, indent=2) + "\n")
    temporary.replace(path)
    return current


def descendants(pid, parent):
    for _ in range(64):
        if pid == parent:
            return True
        if pid <= 1:
            return False
        try:
            fields = Path(f"/proc/{pid}/status").read_text().splitlines()
        except FileNotFoundError:
            return True
        pid = int(next(line.split()[1] for line in fields if line.startswith("PPid:")))
    raise RuntimeError("Process ancestry exceeded 64 levels")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=172800)
    parser.add_argument("--handoff-reservation-pid", type=int)
    parser.add_argument("--contention-policy", choices=("abort", "mark"), default="abort")
    args = parser.parse_args()
    handoff = args.handoff_reservation_pid
    handoff_start = None
    if handoff is not None:
        proc = Path(f"/proc/{handoff}")
        if proc.stat().st_uid != os.getuid() or b"decode_capacity_guard.py" not in (proc / "cmdline").read_bytes():
            raise ValueError("Handoff must name this user's existing reservation guard")
        handoff_start = (proc / "stat").read_text().split(")", 1)[1].split()[19]
    handoff_deadline = time.monotonic() + 60

    def check():
        nonlocal handoff
        if handoff is not None:
            proc = Path(f"/proc/{handoff}")
            try:
                current_start = (proc / "stat").read_text().split(")", 1)[1].split()[19]
            except FileNotFoundError:
                handoff = None
            else:
                if current_start != handoff_start:
                    raise RuntimeError("Reservation handoff PID was reused")
                if time.monotonic() > handoff_deadline:
                    raise RuntimeError("Reservation handoff did not finish within 60 seconds")
        text = subprocess.check_output([
            "nvidia-smi", "-i", os.environ["CUDA_VISIBLE_DEVICES"],
            "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, timeout=15)
        return [int(pid) for pid in text.split() if int(pid) != handoff and not descendants(int(pid), args.parent)]

    if check():
        raise RuntimeError("GPU busy before reservation")
    import torch
    allocations = [torch.zeros(1024, device=f"cuda:{i}") for i in range(torch.cuda.device_count())]
    for i in range(len(allocations)):
        torch.cuda.synchronize(i)
    if check():
        raise RuntimeError("GPU ownership changed while attaching reservation")
    args.ready.write_text(json.dumps({"pid": os.getpid(), "devices": os.environ["CUDA_VISIBLE_DEVICES"]}))
    deadline = time.monotonic() + args.max_seconds
    contention_path = args.ready.with_suffix(".contention.json")
    episodes, active_pids = [], ()
    while time.monotonic() < deadline:
        os.kill(args.parent, 0)
        foreign = check()
        if foreign:
            if args.contention_policy == "abort":
                contention_path.write_text(json.dumps({"foreign_pids": foreign}))
                os.kill(args.parent, signal.SIGTERM)
                raise RuntimeError(f"External GPU contention: {foreign}")
            active_pids = record_contention(contention_path, foreign, episodes, active_pids)
        else:
            active_pids = ()
        time.sleep(5)
    os.kill(args.parent, signal.SIGTERM)
    raise RuntimeError("Reservation lifetime exceeded")


if __name__ == "__main__":
    main()
