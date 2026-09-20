"""Export validated decode samples into the existing standalone plot schema."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--summary', type=Path, required=True)
parser.add_argument('--template', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
summary = json.loads(args.summary.read_text())
template = json.loads(args.template.read_text())
if summary['status'] != 'success' or summary['metric'] != 'decode_stage_throughput_tps':
    raise ValueError('Require a validated decode-only summary')
protocol = {**template['protocol'], 'discard_full_decode_steps': 8,
            'measured_decode_steps': 256, 'cuda_sync': 'window boundaries only',
            'timing_scope': summary['scope'], 'vortex_overlap_schedule': True}
cases = []
for case in summary['cases']:
    selected = [row for row in summary['samples'] if all(row[key] == case[key]
                 for key in ('model', 'method', 'engine', 'concurrency'))]
    if len(selected) != 3:
        raise ValueError(f'Invalid sample count: {case}')
    series = ('Vortex' if case['engine'].startswith('Vortex') else
              'SparseEngine (wave2)' if 'wave2' in case['engine'] else 'SparseEngine')
    cases.append({'model': case['model'],
                  'method': 'H2O / H2O-like' if case['method'] == 'H2O' else case['method'],
                  'series': series, 'concurrency': case['concurrency'],
                  'source': case['source_path'], 'source_sha256': selected[0]['source_sha256'],
                  'timing_source': selected[0]['scope'],
                  'samples': [{'iteration': row['iteration'], 'status': row['status'],
                               'decode_tokens': row['decode_stage_tokens'],
                               'elapsed_s': row['decode_stage_elapsed_s'],
                               'value': row['decode_stage_throughput_tps'],
                               'decode_steps': row['decode_steps'],
                               'graph_counter_delta': row['graph_counter_delta']}
                              for row in selected]})
payload = {'schema_version': 1, 'metric': summary['metric'], 'unit': 'token/s',
           'aggregation': 'arithmetic mean of three measured decode-only window rates; error bars are sample standard deviation',
           'models': template['models'], 'protocol': protocol,
           'caveats': summary['caveats'] + [
               'QuEST head policies differ, especially MLA; no official algorithm parity claim.',
               'Tested residency bounds are not exhaustive maximum concurrency measurements.'],
           'source_summary': str(args.summary),
           'cases': cases}
with args.output.open('x') as handle:
    json.dump(payload, handle, indent=2)
