"""Run a longest-prompt smoke, then the frozen complete paired cohort."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def main(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    dest = Path(spec['output'])
    dest.mkdir(parents=True, exist_ok=True)
    prepared = Path(spec['prepared'])
    for _ in range(1800):
        status = prepared.with_name('run_status.json')
        if status.exists() and json.loads(status.read_text())['status'] == 'failed':
            raise RuntimeError(f'Preparation failed: {status}')
        if prepared.exists():
            break
        time.sleep(2)
    else:
        raise TimeoutError('Prepared cohort was not available within one hour')
    payload = json.loads(prepared.read_text())
    selected = payload['samples']
    expected_count = payload['identity']['token_buckets'][0]['samples']
    if len(selected) != expected_count or len({x['sample']['_id'] for x in selected}) != expected_count:
        raise ValueError('Prepared cohort count/identity mismatch')
    smoke = {**payload, 'identity': {**payload['identity'], 'token_buckets': [
        {**payload['identity']['token_buckets'][0], 'samples': 1}]},
        'samples': [{**max(selected, key=lambda x: x['prompt_tokens']), 'index': 0}]}
    smoke_path = dest / 'smoke_samples.json'
    smoke_path.write_text(json.dumps(smoke))
    for phase, source, count in [('smoke', smoke_path, 1), ('full', prepared, expected_count)]:
        output = dest / phase
        if phase == 'full' and not (dest / 'smoke' / 'run_status.json').exists():
            raise RuntimeError('Missing smoke status')
        buckets = [{**payload['identity']['token_buckets'][0], 'samples': count}]
        cmd = spec['command'] + ['--prepared-samples', str(source), '--output-dir', str(output),
                                '--token-buckets-json', json.dumps(buckets)]
        output.mkdir(exist_ok=False)
        (output / 'command.json').write_text(json.dumps(cmd, indent=2))
        with (output / 'run.log').open('w') as log:
            result = subprocess.run(cmd, cwd=spec['cwd'], stdout=log, stderr=subprocess.STDOUT,
                                    timeout=spec.get('timeout', 36000))
        with (dest / 'status.tsv').open('a') as f:
            f.write(f'{phase}\t{result.returncode}\t{time.time()}\t{output}\n')
        if result.returncode:
            raise RuntimeError(f'{phase} failed with exit {result.returncode}: {output / "run.log"}')
        metrics = json.loads((output / 'aggregate_metrics.json').read_text())
        rows = [json.loads(x) for x in (output / 'sample_results.jsonl').read_text().splitlines()]
        if metrics['status'] != 'success' or metrics['samples'] != count or len(rows) != count:
            raise RuntimeError(f'{phase} incomplete or execution-failed')
        if phase == 'full' and {x['_id'] for x in rows} != {x['sample']['_id'] for x in selected}:
            raise RuntimeError('Paired cohort ID mismatch')
        print(phase, metrics, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec', type=Path)
    main(parser.parse_args().spec)
