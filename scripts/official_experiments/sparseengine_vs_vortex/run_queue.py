"""Bounded, GPU-guarded command queue; timings belong to existing probes."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import shutil
import time

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--gpu', required=True)
p.add_argument('--native-env', type=Path, required=True)
p.add_argument('--conda', default=shutil.which('conda'))
a = p.parse_args()
if not a.conda or not a.native_env.is_dir():
    raise ValueError('A conda executable and existing native environment are required')
if (a.root / 'guard.json').exists():
    raise FileExistsError('Use a fresh queue root; a guard record already exists')
# The guard must never signal an unrelated interactive shell process group.
if os.getpgrp() != os.getpid():
    os.setpgrp()
a.root.mkdir(parents=True, exist_ok=True)
(a.root / 'jobs').mkdir(exist_ok=True)
os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
conda = a.conda
guard_cmd = [conda, 'run', '--no-capture-output', '-p', str(a.native_env.resolve()), 'python', str(Path(__file__).with_name('gpu_guard.py')), '--parent-pid', str(os.getpid()), '--ready', str(a.root / 'guard.json'), '--max-seconds', '43200']
guard_log = (a.root / 'guard.log').open('a')
guard = subprocess.Popen(guard_cmd, stdout=guard_log, stderr=subprocess.STDOUT)
child = None
def stop_child():
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
def interrupted(signum, frame):
    raise KeyboardInterrupt(f'signal {signum}')
signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
try:
    for _ in range(60):
        if (a.root / 'guard.json').exists():
            break
        if guard.poll() is not None:
            raise RuntimeError('GPU guard failed')
        time.sleep(1)
    else:
        raise TimeoutError('GPU guard did not attach')
    seen = set()
    idle_since = time.monotonic()
    deadline = time.monotonic() + 42000
    while time.monotonic() < deadline:
        if guard.poll() is not None:
            raise RuntimeError('GPU guard exited')
        jobs = sorted(x for x in (a.root / 'jobs').glob('*.json') if x.name not in seen)
        if not jobs:
            if (a.root / 'STOP').exists():
                break
            if time.monotonic() - idle_since > 1800:
                raise TimeoutError('No new command for 30 minutes; release GPU')
            time.sleep(2)
            continue
        job = jobs[0]
        spec = json.loads(job.read_text())
        stage = a.root / job.stem
        stage.mkdir(exist_ok=False)
        (stage / 'command.json').write_text(json.dumps(spec, indent=2))
        env = {**os.environ, **spec.get('env', {})}
        for key in ('TMPDIR', 'XDG_CACHE_HOME', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR'):
            env.setdefault(key, str(a.root / 'cache' / key.lower()))
            Path(env[key]).mkdir(parents=True, exist_ok=True)
        begin = time.time()
        with (stage / 'run.log').open('w') as log:
            child = subprocess.Popen(spec['command'], cwd=spec['cwd'], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            (stage / 'running.json').write_text(json.dumps({'pid': child.pid, 'start': begin}))
            timed_out = False
            while child.poll() is None:
                if guard.poll() is not None:
                    raise RuntimeError('Guard exited during command')
                if time.time() - begin > spec.get('timeout', 3600):
                    timed_out = True
                    stop_child()
                    break
                time.sleep(2)
        result = {'exit_code': child.returncode, 'timeout': timed_out, 'start': begin, 'finish': time.time()}
        (stage / 'exit.json').write_text(json.dumps(result, indent=2))
        with (a.root / 'status.tsv').open('a') as status:
            status.write(f'{job.stem}\t{child.returncode}\t{time.time()}\t{stage}\n')
        print(job.stem, result, flush=True)
        seen.add(job.name)
        idle_since = time.monotonic()
        if child.returncode != 0 and not (spec.get('capacity_probe', False) or spec.get('continue_on_failure', False)):
            raise RuntimeError(f'{job.stem} failed; queue stopped')
    else:
        raise TimeoutError('Queue lifetime exceeded')
finally:
    stop_child()
    if (a.root / 'guard.json').exists():
        guard_pid = json.loads((a.root / 'guard.json').read_text())['pid']
        proc = Path(f'/proc/{guard_pid}/cmdline')
        if proc.exists() and b'gpu_guard.py' in proc.read_bytes():
            os.kill(guard_pid, signal.SIGTERM)
    try:
        guard.wait(timeout=30)
    except subprocess.TimeoutExpired:
        guard.terminate()
    (a.root / 'queue_exit.json').write_text(json.dumps({'finished': time.time()}))
