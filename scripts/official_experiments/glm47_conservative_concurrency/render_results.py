"""Render the recorded aggregates; never derive missing measurement fields."""
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    rows = json.loads((root / 'results.json').read_text())
    keys = [(r['device'], r['input_tokens'], r['method']) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate experiment points')
    fields = ['device', 'weight_dtype', 'input_tokens', 'method', 'concurrency',
              'decode_tps', 'ttft_s', 'tpot_ms', 'e2e_s', 'output_tps']
    lines = ['# GLM-4.7-Flash conservative concurrency results', '',
             'Decode: full-batch continuous-window tokens/s. TTFT, TPOT and E2E: request means.',
             'Output TPS: all output tokens / complete measured workload time.',
             'Concurrency is verified conservatively, not a proven maximum. Missing exports are shown as —.', '',
             '| Device | Weights | Input tokens | Method | Concurrency | Decode tok/s | TTFT s | TPOT ms | E2E s | Output tok/s |',
             '|---|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        cells = []
        for field in fields:
            value = row[field]
            cells.append('—' if value is None else f'{value:.2f}' if isinstance(value, float) else str(value))
        lines.append('| ' + ' | '.join(cells) + ' |')
    (root / 'RESULTS.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
