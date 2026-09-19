"""Own exactly one external server while the canonical quality runner evaluates it."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import urlopen
from urllib.error import URLError

def main(spec_path):
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    spec = json.loads(Path(spec_path).read_text())
    dest = Path(spec['output'])
    dest.mkdir(parents=True, exist_ok=True)
    with (dest/'server.log').open('w') as log:
        server = subprocess.Popen(spec['server_command'], cwd=spec['server_cwd'],
                                  env={**os.environ,**spec.get('server_env',{})},stdout=log,
                                  stderr=subprocess.STDOUT,start_new_session=True)
        (dest/'server_pid.json').write_text(json.dumps({'pid':server.pid,'command':spec['server_command']}))
        try:
            for _ in range(600):
                if server.poll() is not None:
                    raise RuntimeError(f'Server exited before readiness: {server.returncode}')
                try:
                    with urlopen(spec['server_url']+'/health',timeout=2) as response:
                        if response.status==200: break
                except (URLError,TimeoutError):
                    pass
                time.sleep(2)
            else:
                raise TimeoutError('Server readiness deadline exceeded')
            result = subprocess.run(['python3',str(Path(__file__).with_name('run_quality.py')),spec['quality_spec']],timeout=39000)
            if result.returncode: raise RuntimeError(f'Quality exited {result.returncode}')
        finally:
            if server.poll() is None:
                os.killpg(server.pid,signal.SIGTERM)
                try: server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid,signal.SIGKILL)
                    server.wait(timeout=30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec', type=Path)
    main(parser.parse_args().spec)
