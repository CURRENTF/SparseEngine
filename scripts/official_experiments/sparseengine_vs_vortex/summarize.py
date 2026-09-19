"""Validate matched traces and synchronized decode-only windows, not request proxies."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from benchmark.efficiency.metrics import stage_throughput

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--cases', type=Path, default=Path(__file__).with_name('configs') / 'cases.json')
args = parser.parse_args()
manifest = json.loads(args.cases.read_text())
cases = [(c['model'], c['method'], c['engine'], c['directory'], c['concurrency'])
         for c in manifest['cases'] if c['phase'] == 'measure']
summary, all_samples, traces = [], [], {}
for model, method, engine, relative, batch in cases:
    case = args.root / relative
    vortex = engine.startswith('Vortex')
    probe = case / 'measured' if vortex else case
    if json.loads((case / 'exit.json').read_text())['exit_code'] != 0:
        raise RuntimeError(f'Failed case: {case}')
    if json.loads((probe / 'run_status.json').read_text())['status'] != 'success':
        raise RuntimeError(f'Failed probe: {probe}')
    records = [json.loads(line) for line in (probe / 'raw_samples.jsonl').read_text().splitlines()]
    requests = [json.loads(line) for line in (probe / 'request_samples.jsonl').read_text().splitlines()]
    if len(records) != 3 or len(requests) != batch * 3 or any(
            r['status'] != 'success' or r['generated_tokens'] != 512 for r in requests):
        raise RuntimeError(f'Incomplete measured requests: {probe}')
    trace = [[(r['prompt_digest'], r['prompt_len'], r['output_len'])
              for r in record['trace']['requests']] for record in records]
    key = (model, method, batch)
    if key in traces and traces[key] != trace:
        raise RuntimeError(f'Mismatched comparison trace: {key}')
    traces[key] = trace
    if vortex:
        source = case / 'decode_windows_labeled.json'
        windows = [row for row in json.loads(source.read_text()) if row['phase'] == 'measure']
    else:
        source = probe / 'raw_samples.jsonl'
        windows = [row['decode_only_window'] for row in records]
    if len(windows) != 3:
        raise RuntimeError(f'Expected three measured windows: {case}')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    rates = []
    for iteration, (window, record) in enumerate(zip(windows, records)):
        expected_graph = {'capture_count': 0, 'replay_count': 256, 'eager_decode_count': 0}
        if (window['status'] != 'success' or window['concurrency'] != batch
                or window['decode_steps'] != 256 or window['discarded_full_decode_steps'] != 8
                or window['decode_stage_tokens'] != batch * 256 or window['prefill_steps'] != 0
                or window['graph_counter_delta'] != expected_graph):
            raise RuntimeError(f'Invalid decode window: {case}, {iteration}')
        rate = stage_throughput(window['decode_stage_tokens'], window['decode_stage_elapsed_s'])
        if not math.isclose(rate, window['decode_stage_throughput_tps'], rel_tol=1e-12):
            raise RuntimeError(f'Inconsistent decode arithmetic: {case}, {iteration}')
        if vortex:
            # perf_counter is a shared monotonic host clock across these processes.
            cohort = [r for r in requests if r['iteration'] == iteration]
            if (max(r['first_token_at_s'] for r in cohort) > window['started_monotonic_s']
                    or min(r['finished_at_s'] for r in cohort) < window['finished_monotonic_s']):
                raise RuntimeError(f'Window does not lie inside all live decode requests: {case}')
        elif record['peak_scheduler_decoding_requests'] != batch:
            raise RuntimeError(f'Native residency mismatch: {case}')
        rates.append(rate)
        all_samples.append({**window, 'model': model, 'method': method, 'engine': engine,
                            'iteration': iteration, 'source_path': str(source),
                            'source_sha256': digest})
    summary.append(dict(model=model, method=method, engine=engine, concurrency=batch,
                        mean_tps=statistics.fmean(rates), sd_tps=statistics.stdev(rates),
                        pooled_tps=stage_throughput(sum(w['decode_stage_tokens'] for w in windows),
                            sum(w['decode_stage_elapsed_s'] for w in windows)),
                        source_path=str(source), status='success'))
result = {'status': 'success', 'metric': 'decode_stage_throughput_tps',
          'scope': 'full-residency continuous decode-only engine window; 8 discard + 256 measured; boundary sync only',
          'caveats': ['H2O-like is not algorithm-equivalent to native H2O; no quality claim',
                      'High concurrency uses a different workload size, wave2 admission excluded',
                      'Not isolated GPU kernel time; request metrics are boundary-perturbed'],
          'cases': summary, 'samples': all_samples}
with args.output.open('x') as handle:
    json.dump(result, handle, indent=2)
for row in summary:
    print(row['model'], row['method'], row['engine'], row['concurrency'],
          f"{row['mean_tps']:.2f} ± {row['sd_tps']:.2f}")
