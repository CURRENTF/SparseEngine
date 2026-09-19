"""Server lifecycle and capacity selection; canonical probe owns all timing."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import psutil

p = argparse.ArgumentParser()
p.add_argument('--config', type=Path, required=True)
a = p.parse_args()
c = json.loads(a.config.read_text())
root = Path(c['output_dir'])
root.mkdir(parents=True, exist_ok=True)
def interrupted(sig, frame):
    raise KeyboardInterrupt(f'signal {sig}')
signal.signal(signal.SIGTERM, interrupted)
server = None
def get(path):
    with urllib.request.urlopen(c['url'] + path, timeout=10) as response:
        return json.load(response)
def run_probe(name, prompt, output, bs, warmup, iters):
    out = root / name
    out.mkdir(exist_ok=False)
    cmd = [sys.executable, '-u', 'benchmark/efficiency/bench_probe.py', '--server-url', c['url'], '--model-path', c['model_path'], '--backend-label', c['label'], '--sparse-method', c['method'], '--prompt-lens', str(prompt), '--output-lens', str(output), '--batch-sizes', str(bs), '--scenario', 'fixed', '--seed', '42', '--prompt-length-jitter', '0', '--output-length-jitter', '0', '--num-warmups', str(warmup), '--num-iters', str(iters), '--monitor-gpus', str(c['gpu']), '--request-timeout-s', '1800', '--output-dir', str(out)]
    (out / 'command.json').write_text(json.dumps(cmd, indent=2))
    with (out / 'run.log').open('w') as log:
        subprocess.run(cmd, cwd=c['vortex_repo'], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=7200)
    status = json.loads((out / 'run_status.json').read_text())
    rows = [json.loads(x) for x in (out / 'request_samples.jsonl').read_text().splitlines()]
    if status['status'] != 'success' or len(rows) != bs * iters or any(r['status'] != 'success' or r['generated_tokens'] != output for r in rows):
        raise RuntimeError(f'{name}: incomplete benchmark outputs')
    print(name, 'success', len(rows), flush=True)
try:
    with (root / 'server.log').open('w') as log:
        server = subprocess.Popen([sys.executable, '-u', str(Path(__file__).with_name('vortex_server.py')), str(a.config)], stdout=log, stderr=subprocess.STDOUT, cwd=c['vortex_repo'])
        for _ in range(600):
            if server.poll() is not None:
                raise RuntimeError(f'Server exited {server.returncode}; see server.log')
            try:
                with urllib.request.urlopen(c['url'] + '/health', timeout=10) as response:
                    if response.status != 200:
                        raise RuntimeError('Unexpected health response')
                info = get('/get_server_info')
                break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(2)
        else:
            raise TimeoutError('Server readiness exceeded 20 minutes')
        (root / 'server_info.json').write_text(json.dumps(info, indent=2))
        print('server_ready', flush=True)
        run_probe('smoke', 8192, 16, 1, 0, 1)
        if c.get('smoke_only', False):
            print('smoke_only_complete', flush=True)
        else:
            states = info['internal_states']
            capacity = min(int(s['memory_usage']['token_capacity']) for s in states)
            max_running = min(int(s['effective_max_running_requests_per_dp']) for s in states)
            available_bs = min(c['max_running_requests'], max_running, capacity // (32768 + 512 + 16))
            bs = int(c['expected_concurrency'])
            if bs < 1 or available_bs < bs:
                raise RuntimeError(f'Recorded concurrency {bs} does not fit; capacity supports {available_bs}. Refusing to change the experiment.')
            (root / 'capacity.json').write_text(json.dumps({'max_total_num_tokens': capacity, 'selected_batch_size': bs, 'reserved_tokens_per_request': 32768 + 512 + 16, 'scope': 'KV residency upper bound; verify scheduler running count in server log'}, indent=2))
            run_probe('measured', 32768, 512, bs, 1, 3)
            if os.environ.get('PAPER_DECODE_WINDOW_OUTPUT'):
                windows = [json.loads(line) for line in Path(os.environ['PAPER_DECODE_WINDOW_OUTPUT']).read_text().splitlines()]
                if len(windows) != 7 or any(row['concurrency'] != bs for row in windows):
                    raise RuntimeError(f'Expected warmup plus three preparation/measurement window pairs; got {len(windows)}')
                roles = [('warmup', 0), ('prepare', 0), ('measure', 0), ('prepare', 1), ('measure', 1), ('prepare', 2), ('measure', 2)]
                for row, (phase, iteration) in zip(windows, roles):
                    row.update(phase=phase, iteration=iteration)
                (root / 'decode_windows_labeled.json').write_text(json.dumps(windows, indent=2))
finally:
    if server is not None:
        try:
            processes = psutil.Process(server.pid).children(recursive=True)
            for process in reversed(processes):
                try:
                    process.terminate()
                except psutil.NoSuchProcess:
                    pass
            if server.poll() is None:
                server.terminate()
            _, alive = psutil.wait_procs(processes, timeout=20)
            for process in alive:
                process.kill()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
        except psutil.NoSuchProcess:
            server.wait(timeout=20)
