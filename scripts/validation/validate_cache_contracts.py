"""Execute the cache contract gates; never report unavailable hardware as a pass.

Run from a repository checkout with its normal test dependencies installed:
    python scripts/validation/validate_cache_contracts.py --tier cpu
    python scripts/validation/validate_cache_contracts.py --tier cuda
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
CPU = (
    'tests/test_cache_contract_oracle.py',
    'tests/test_chain_cache_contract.py',
    'tests/test_radix_cache_contract.py',
    'tests/test_cache_execution_contract.py',
    'tests/test_cache_batches.py',
    'tests/test_cache_callpaths.py',
    'tests/test_cache_pressure.py',
)
REGRESSION = (
    'tests/test_chain_capacity_reservations.py',
    'tests/test_chain_prefill_capacity.py',
    'tests/test_decode_idle_chain_reclaim.py',
    'tests/test_chain_prefix_cache.py',
    'tests/test_chain_offload.py',
    'tests/test_prefix_cache.py',
    'tests/test_async_scheduler.py',
)
CUDA = (
    'tests/test_cache_contract_cuda.py',
    'tests/test_async_execution_cuda.py',
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tier', choices=('cpu', 'regression', 'cuda'), default='cpu')
    parser.add_argument('--output', type=Path, default=ROOT / 'cache-contract-results')
    args = parser.parse_args()
    if args.tier == 'cuda':
        import torch
        if not torch.cuda.is_available():
            parser.error('CUDA tier requires a real GPU; this is not a passing skip.')
    targets = {'cpu': CPU, 'regression': REGRESSION, 'cuda': CUDA}[args.tier]
    missing = [target for target in targets if not (ROOT / target).is_file()]
    if missing:
        parser.error(f'Missing required contract suites: {missing}')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    junit = out / f'{args.tier}.xml'
    if junit.exists():
        junit.unlink()  # Never reuse a previous successful run's report.
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join((str(ROOT / 'src'), str(ROOT / 'tests'), env.get('PYTHONPATH', '')))
    command = [sys.executable, '-m', 'pytest', '-q', '--strict-markers']
    if args.tier == 'regression':
        command.extend(('-m', 'not cuda'))
    command.extend((f'--junitxml={junit}', *targets))
    result = subprocess.run(command, cwd=ROOT, env=env, check=False)
    stats = {'tests': 0, 'failures': 0, 'errors': 0, 'skipped': 0}
    if junit.exists():
        root = ET.parse(junit).getroot()
        suites = [root] if root.tag == 'testsuite' else root.findall('testsuite')
        for suite in suites:
            for name in stats:
                stats[name] += int(suite.get(name, 0))
    # CUDA cases in legacy regression files are deselected above. Every selected
    # case is required; an unavailable dependency must not become a passing skip.
    ok = (result.returncode == 0 and stats['tests'] > 0
          and stats['failures'] == stats['errors'] == 0
          and stats['skipped'] == 0)
    report = {'tier': args.tier, 'passed': ok, 'python': sys.version,
              'command': command, **stats}
    (out / f'{args.tier}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if ok else (result.returncode or 1)


if __name__ == '__main__':
    raise SystemExit(main())
