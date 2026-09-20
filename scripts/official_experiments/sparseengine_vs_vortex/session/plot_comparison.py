"""Render the available campaign data without treating missing runs as successes."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import seaborn as sns


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--summary-dir', type=Path, help='Validated summary directory; defaults to ROOT/summary')
    parser.add_argument('--old-data', type=Path, required=True)
    parser.add_argument('--sm-data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary_dir = args.summary_dir or args.root / 'summary'
    repo = Path(__file__).resolve().parents[4]
    capacity = module(repo / 'scripts/official_experiments/sparse_decode_efficiency/plot_decode_capacity.py', 'capacity')
    old_plot = module(repo / 'scripts/official_experiments/sparseengine_vs_vortex/plot.py', 'old_plot')
    sources = {}

    def read(path):
        path = Path(path)
        raw = path.read_bytes()
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    quality = read(summary_dir / 'quality.json')
    for q in quality:
        if q['status'] == 'completed':
            aggregate = read(Path(q['artifact']) / 'aggregate_metrics.json')
            assert aggregate['samples'] == q['evaluated_samples'] == 198
            assert math.isclose(aggregate['correct'] / 198 * 100, q['accuracy'])
            assert aggregate['accuracy'] == q['accuracy']

    def qvalue(prefix):
        found = [q for q in quality if q['case'].startswith(prefix)]
        assert len(found) == 1, prefix
        q = found[0]
        return f"{q['accuracy']:.2f}" if q['status'] == 'completed' else '—'

    data32 = read(summary_dir / 'comparison32.json')
    data128 = read(summary_dir / 'comparison128.json')
    for rows, config in [(data32, dict(input_len=32768, output_len=512)),
                         (data128, dict(input_len=131072, output_len=2048))]:
        for row in rows:
            verified = capacity.validate_measurement(Path(row['artifact']), row['concurrency'], config)
            assert verified['decode_throughput_tps'] == row['decode_throughput_tps']
    if not data32 or not data128:
        raise ValueError('Both efficiency groups require at least one validated point')
    print(f'Validated {len(data32) + len(data128)} raw efficiency points and '
          f'{sum(q["status"] == "completed" for q in quality)} quality aggregates.', flush=True)

    old = read(args.old_data)
    sm = read(args.sm_data)
    for case in old['cases']:
        source = Path(case['source'])
        assert hashlib.sha256(source.read_bytes()).hexdigest() == case['source_sha256']
        if case['model'] == 'GLM-4.7-Flash' and case['method'].startswith('H2O') and case['series'] != 'Vortex':
            matched = [s for s in sm['samples'] if s['engine'] == 'SM Triton' and s['concurrency'] == case['concurrency']]
            assert len(matched) == 3
            for s in matched:
                assert hashlib.sha256(Path(s['source_path']).read_bytes()).hexdigest() == s['source_sha256']
                assert s['status'] == 'success' and s['decode_steps'] == 256
            case['source'] = matched[0]['source_path']
            case['source_sha256'] = matched[0]['source_sha256']
            case['samples'] = [dict(iteration=s['iteration'], status=s['status'], decode_tokens=s['decode_stage_tokens'],
                                    elapsed_s=s['decode_stage_elapsed_s'], value=s['decode_stage_throughput_tps']) for s in matched]
    old_plot.validated_rows(old)
    args.output.mkdir(parents=True, exist_ok=False)
    sns.set_theme(style='whitegrid', context='paper', font='DejaVu Sans', font_scale=1.15,
                  rc={'axes.spines.top': False, 'axes.spines.right': False, 'grid.alpha': .3,
                      'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    palette = read(repo / 'scripts/official_experiments/sparse_decode_efficiency/palettes/fresh_modern.json')['colors']
    lanes = {**capacity.LANES, **capacity.EXTERNAL_LANES}
    names = {k: (('Ours (' + v[0] + ')') if k.startswith('sengine') else v[0]) for k, v in lanes.items()}
    names['hisparse-quest'] += ' †'
    outputs = []

    def save(fig, name):
        for ext in ('png', 'pdf', 'svg'):
            target = args.output / f'{name}.{ext}'
            fig.savefig(target, dpi=220, bbox_inches='tight', facecolor='white')
            outputs.append(str(target))
        plt.close(fig)

    def table(ax, rows):
        ax.axis('off')
        ax.text(0, 1.02, 'LongBench v2 medium', weight='bold', fontsize=11, transform=ax.transAxes)
        ax.text(0, .94, 'Common 198 samples · accuracy (%)', fontsize=9, color='#667085', transform=ax.transAxes)
        tab = ax.table(cellText=[[name, value] for name, value, color in rows],
                       colLabels=['Method', 'Quality ↑'], colWidths=[.77, .23],
                       cellLoc='left', bbox=[0, .08, 1, .79])
        tab.auto_set_font_size(False)
        tab.set_fontsize(9.5)
        for (i, j), cell in tab.get_celld().items():
            cell.set_edgecolor('#E5E9EF')
            cell.set_linewidth(.55)
            if i == 0:
                cell.set_facecolor('#EFF3F7')
                cell.get_text().set_weight('bold')
            else:
                cell.set_facecolor('#FAFBFC' if i % 2 else 'white')
                if j == 0:
                    cell.get_text().set_color(rows[i-1][2])
            if j == 1:
                cell.get_text().set_ha('right')

    def curves(ax, rows, title, ticks=None):
        for lane in lanes:
            points = sorted([r for r in rows if r['lane'] == lane], key=lambda r: r['concurrency'])
            if not points:
                continue
            ax.plot([r['concurrency'] for r in points], [r['decode_throughput_tps'] for r in points],
                    color=palette[lane], marker=lanes[lane][1], label=names[lane], markersize=5, linewidth=1.7)
        ax.set_xscale('log', base=2)
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
        if ticks:
            ax.set_xticks(ticks)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Batch size B')
        ax.set_ylabel('Decode throughput (token/s)')
        ax.set_title(title, loc='left', fontsize=11, pad=12)

    def curve_quality(rows, include_unsupported=False):
        result = []
        for lane in lanes:
            matches = [r for r in rows if r['lane'] == lane]
            if matches:
                value = matches[0]['quality_accuracy']
                result.append((names[lane], '—' if value is None else f'{value:.2f}', palette[lane]))
            elif include_unsupported and lane in capacity.EXTERNAL_LANES:
                result.append((names[lane], 'N/A', palette[lane]))
        return result

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.9), gridspec_kw={'width_ratios': [1.5, 1]})
    for i, (model, title) in enumerate([('qwen3-30b-fp8', 'Qwen3-30B-A3B FP8 · TP1/EP1'),
                                       ('glm4.7-flash', 'GLM-4.7-Flash BF16 · TP2/EP2')]):
        rows = [r for r in data128 if r['model'] == model]
        curves(axes[i, 0], rows, title)
        table(axes[i, 1], curve_quality(rows, include_unsupported=i == 1))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False, fontsize=9)
    fig.subplots_adjust(top=.85, bottom=.16, hspace=.64, wspace=.29)
    fig.text(.06, .055, 'Efficiency: 128K input / 2K output; synchronized full-batch decode steps. Quality: TP1 for all models.\n'
             '† HiSparse quality uses a fixed ratio, not a fixed token budget. — incomplete; N/A unsupported MLA.', fontsize=9)
    save(fig, 'capacity_128k_quality')

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), gridspec_kw={'width_ratios': [1.5, 1]})
    curves(axes[0], data32, 'Qwen3-4B BF16 · 32K input / 512 output · TP1', [1, 4, 8, 14])
    table(axes[1], curve_quality(data32))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False, fontsize=9)
    fig.subplots_adjust(top=.78, bottom=.28, wspace=.29)
    missing = []
    for lane, label in [('hisparse-quest', 'HiSparse'), ('sengine-quest', 'Ours QuEST'),
                        ('tangram-snapkv', 'Tangram'), ('sengine-snapkv', 'Ours SnapKV')]:
        batches = {r['concurrency'] for r in data32 if r['lane'] == lane}
        absent = [str(b) for b in [1, 4, 8, 14] if b not in batches]
        if absent:
            missing.append(label + ' B' + '/'.join(absent))
    coverage = f'{len(data32)}/16 valid points.' + (' Missing: ' + ', '.join(missing) + '.' if missing else '')
    fig.text(.06, .065, 'Synchronized full-batch decode steps; ' + coverage + '\n'
             '† HiSparse quality uses a fixed ratio. GLM: Tangram / HiSparse do not support MLA.', fontsize=9)
    save(fig, 'new_baselines_32k_quality')

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.9), gridspec_kw={'width_ratios': [1.5, 1]})
    series = list(old_plot.PALETTE)
    for i, (model, prefix) in enumerate([('Qwen3-4B', 'qwen4'), ('GLM-4.7-Flash', 'glm')]):
        ax = axes[i, 0]
        for j, engine in enumerate(series):
            for x, method in enumerate(['QuEST', 'H2O / H2O-like']):
                matches = [c for c in old['cases'] if c['model'] == model and c['method'] == method and c['series'] == engine]
                if not matches:
                    continue
                c, = matches
                values = [s['value'] for s in c['samples']]
                pos = x + (j-1)*.25
                rect = ax.bar(pos, statistics.mean(values), .22, yerr=statistics.stdev(values),
                              capsize=2, color=old_plot.PALETTE[engine], label=engine if x == 1 else None)
                ax.bar_label(rect, labels=[f'{statistics.mean(values):.0f}\nB={c["concurrency"]}'], padding=4, fontsize=9)
        ax.set_xticks([0, 1], ['QuEST', 'H2O'])
        ax.set_ylim(0, ax.get_ylim()[1]*1.22)
        ax.set_ylabel('Decode-window throughput (token/s)')
        ax.set_title(model + ' · BF16 · TP1', loc='left', fontsize=11, pad=12)
        rows = [('Ours (Vanilla)', qvalue(prefix+'_vanilla_32k'), '#70838A')]
        for method in ('quest', 'h2o'):
            label = 'QuEST' if method == 'quest' else 'H2O'
            rows.append((f'Ours ({label})', qvalue(f'{prefix}_{method}_32k'), '#329A89'))
            value = qvalue(f'{prefix}_vortex_{method}_launchfix')
            if prefix == 'qwen4' and method == 'quest':
                value += ' *'
            rows.append((f'Vortex ({label})', value, '#658EB6'))
        table(axes[i, 1], rows)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, [s.replace('SparseEngine', 'Ours').replace('wave2', 'Max Avai. B') for s in labels],
               loc='upper center', ncol=3, frameon=False)
    fig.subplots_adjust(top=.87, bottom=.20, hspace=.60, wspace=.29)
    fig.text(.06, .045, '32K / 512; 256-step boundary-synchronized windows, mean ± SD (3 runs). GLM H2O uses the SM-fix rerun.\n'
             '* Vortex QuEST: 197/198 parse failures; anomalous output, not a reliable method-quality ranking. — incomplete.\n'
             'MLA H2O remains approximate. Max Avai. B is a tested residency bound; quality is evaluated once per method.', fontsize=8.7)
    save(fig, 'vortex_32k_quality')

    (args.output / 'plot_data.json').write_text(json.dumps(dict(quality=quality, efficiency32=data32,
        efficiency128=data128, boundary32=old, status='partial_preview'), indent=2)+'\n')
    (args.output / 'manifest.json').write_text(json.dumps(dict(command=sys.argv, source_sha256=sources,
        validation=f'{len(data32) + len(data128)} stage points revalidated from raw steps/full outputs; quality aggregates and boundary source hashes checked',
        outputs=outputs, matplotlib=matplotlib.__version__, seaborn=sns.__version__), indent=2)+'\n')
    print('\n'.join(outputs), flush=True)


if __name__ == '__main__':
    main()
