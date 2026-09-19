"""Hold a task-owned CUDA context between paired lanes; never evict other jobs."""
import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path
import torch

p=argparse.ArgumentParser()
p.add_argument("--ready",type=Path,required=True)
p.add_argument("--parent-pid",type=int,required=True)
p.add_argument("--max-seconds",type=int,default=86400)
a=p.parse_args()
task_group = os.getpgid(a.parent_pid)
if task_group != os.getpgrp():
    raise RuntimeError("Guard must belong to the queue's task process group")

def belongs_to_queue(pid):
    for _ in range(64):
        if pid in (a.parent_pid, os.getpid()):
            return True
        if pid <= 1:
            return False
        try:
            status = Path(f"/proc/{pid}/status").read_text()
        except FileNotFoundError:
            return True  # It exited after the nvidia-smi snapshot.
        pid = int(next(line.split()[1] for line in status.splitlines() if line.startswith("PPid:")))
    raise RuntimeError("Process ancestry exceeds bounded ownership check")

initial = subprocess.check_output([
    "nvidia-smi", "-i", os.environ["CUDA_VISIBLE_DEVICES"],
    "--query-compute-apps=pid", "--format=csv,noheader,nounits",
], text=True, timeout=10)
if any(not belongs_to_queue(int(pid)) for pid in initial.split()):
    raise RuntimeError("GPU became busy before guard allocation; refusing to attach")
allocation=torch.zeros(16*1024*1024,device="cuda",dtype=torch.float32)
torch.cuda.synchronize()
a.ready.write_text(json.dumps({"pid":os.getpid(),"visible_devices":os.environ["CUDA_VISIBLE_DEVICES"],"allocated_bytes":allocation.numel()*allocation.element_size()})+"\n")
deadline = time.monotonic() + a.max_seconds
while time.monotonic() < deadline:
    os.kill(a.parent_pid, 0)
    output = subprocess.check_output([
        "nvidia-smi", "-i", os.environ["CUDA_VISIBLE_DEVICES"],
        "--query-compute-apps=pid", "--format=csv,noheader,nounits",
    ], text=True, timeout=10)
    foreign = [int(pid) for pid in output.split() if not belongs_to_queue(int(pid))]
    if foreign:
        a.ready.with_name("contention.json").write_text(json.dumps({
            "status":"invalid_external_contention", "foreign_pids":foreign,
            "clock_s":time.perf_counter(), "action":"interrupt_only_own_queue_group",
        })+"\n")
        os.killpg(task_group, signal.SIGINT)
        raise RuntimeError("External GPU ownership changed during guarded run")
    a.ready.touch()
    time.sleep(2)
raise RuntimeError("GPU guard reached its bounded lifetime")
