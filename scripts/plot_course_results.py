"""Plot actual training logs and sequence metrics, with source hash provenance.

Example: python scripts/plot_course_results.py --input output/smoke --output output/curves/smoke
No model inference or synthetic data generation is performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter


COLORS = {'single': '#2563eb', 'temporal_0.1': '#0f766e', 'temporal_0.3': '#ea580c',
          'original': '#64748b', 'selected': '#7c3aed'}
METRICS = [('absrel', 'AbsRel (lower is better)'), ('rmse', 'RMSE, metres (lower is better)'),
           ('delta1', 'Delta1 accuracy (higher is better)'),
           ('temporal_error', 'Sequence temporal error (lower is better)'),
           ('scale_drift', 'Scale drift (lower is better)'),
           ('shift_drift', 'Shift drift (lower is better)')]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def label(name, selection):
    known = {'single': 'Single-frame', 'original': 'Original weights',
             'temporal_candidate': 'Temporal candidate',
             'temporal_candidate_smooth': 'Temporal candidate + EMA',
             'temporal': 'Temporal model', 'temporal_smooth': 'Temporal model + EMA',
             'selected': 'Selected: ' + ('single-frame' if selection.get('fallback_to_single') else 'temporal')}
    if name in known:
        if name.startswith('temporal_candidate'):
            candidate = selection.get('temporal_candidate', '')
            for weight in ('0.1', '0.3'):
                if f'temporal_{weight}_' in candidate:
                    return known[name] + f' (lambda={weight})'
        return known[name]
    for family in ('single', 'temporal_0.1', 'temporal_0.3'):
        if name == family or name.startswith(family + '_'):
            title = 'Single-frame' if family == 'single' else 'Temporal lambda=' + family.split('_')[1]
            suffix = name[len(family):].strip('_')
            return title + (f' / {suffix}' if suffix else '')
    return name.replace('_', ' ')


def color(name, selection):
    if name.startswith('temporal_candidate'):
        return color(selection.get('temporal_candidate', 'temporal_0.3'), selection)
    for family in COLORS:
        if name == family or name.startswith(family + '_'):
            return COLORS[family]
    return '#0891b2'


def decorate(figure, title, detail, smoke):
    prefix = 'SMOKE - pipeline verification only\n' if smoke else ''
    figure.suptitle(prefix + title, fontsize=17, fontweight='bold')
    # Put provenance below the figure canvas; bbox_inches='tight' includes it
    # without colliding with tick labels or x-axis titles.
    figure.text(.5, -.035, detail, ha='center', va='top', fontsize=9, color='#475569')


def read_training(log):
    rows = {}
    duplicates = 0
    for line_number, line in enumerate(log.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        update = int(row['update'])
        if update < 1:
            raise ValueError(f'{log}:{line_number}: update must be positive')
        for metric in ('spatial', 'temporal', 'total'):
            if not math.isfinite(float(row[metric])):
                raise ValueError(f'{log}:{line_number}: nonfinite {metric}')
        duplicates += update in rows
        rows[update] = row
    if not rows:
        raise ValueError(f'Empty training log: {log}')
    return [rows[key] for key in sorted(rows)], duplicates


def training_figure(runs, selection, output, smoke):
    figure, axes = plt.subplots(1, 3, figsize=(14, 5.2), constrained_layout=True)
    for axis, metric, title in zip(axes, ('spatial', 'temporal', 'total'),
                                 ('Spatial supervision loss', 'Unweighted temporal loss', 'Total training objective')):
        for name, rows in runs.items():
            axis.plot([r['update'] for r in rows], [r[metric] for r in rows],
                      label=label(name, selection), color=color(name, selection), marker='o',
                      linewidth=1.8, markersize=4)
        axis.set(title=title, xlabel='Optimizer update', ylabel='Loss')
        axis.grid(alpha=.22)
        axis.legend(fontsize=8)
        axis.ticklabel_format(axis='y', style='plain', useOffset=False)
        axis.xaxis.get_major_locator().set_params(integer=True)
    decorate(figure, 'ZipDepth training curves',
             'Actual JSONL records. Total objectives use different temporal weights; compare spatial and unweighted temporal losses separately.', smoke)
    path = output / 'training_loss.png'
    figure.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(figure)
    return path


def metric_figure(records, split, selection, output, smoke):
    names = list(records)
    if not names:
        raise ValueError(f'No {split} metric records')
    figure, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, METRICS):
        values = [float(records[name][metric]) for name in names]
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError(f'Nonfinite or negative {split} metric {metric}')
        bars = axis.barh(range(len(names)), values, color=[color(name, selection) for name in names], height=.65)
        axis.set_yticks(range(len(names)), [label(name, selection) for name in names], fontsize=9)
        axis.invert_yaxis()
        axis.set_title(title, fontsize=11)
        maximum = max(values)
        axis.set_xlim(0, 1.12 if metric == 'delta1' else max(maximum * 1.3, 1e-6))
        axis.grid(axis='x', alpha=.2)
        axis.set_axisbelow(True)
        if metric == 'delta1':
            axis.xaxis.set_major_formatter(PercentFormatter(1))
            axis.set_xticks([0, .25, .5, .75, 1])
        else:
            axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
        for name, bar, value in zip(names, bars, values):
            text = f'{value:.2%}' if metric == 'delta1' else f'{value:.5f}'
            axis.annotate(text, (value, bar.get_y() + bar.get_height() / 2),
                          xytext=(5, 0), textcoords='offset points', va='center', fontsize=9)
            if name == selection.get('chosen') or name == 'selected':
                bar.set_edgecolor('#111827')
                bar.set_linewidth(1.5)
        if split == 'validation' and metric == 'absrel':
            baseline = [records[name]['absrel'] for name in names if name.startswith('single_')]
            if baseline:
                gate = min(baseline) * 1.02
                axis.axvline(gate, color='#b91c1c', linestyle='--', linewidth=1,
                             label='Single-frame AbsRel + 2% gate')
                axis.text(gate, .99, '+2% gate', transform=axis.get_xaxis_transform(),
                          ha='left', va='top', fontsize=8, color='#b91c1c')
    first = records[names[0]]
    sequence_count = len(first.get('sequences', []))
    detail = (f"{split.title()}: {first.get('frames', '?')} unique frames, {first.get('pairs', '?')} adjacent pairs, "
              f"{sequence_count} {'trajectory' if sequence_count == 1 else 'trajectories'}. Temporal metrics use one affine alignment per trajectory. "
              'Outlined bars indicate the selected model.')
    decorate(figure, f'ZipDepth {split} comparison', detail, smoke)
    path = output / f'{split}_metrics.png'
    figure.savefig(path, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(figure)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='output/experiments', help='Experiment directory containing selection.json and run logs')
    parser.add_argument('--output', default='output/curves')
    args = parser.parse_args()
    source, output = Path(args.input).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    selection_path = source / 'selection.json'
    selection = json.loads(selection_path.read_text(encoding='utf-8')) if selection_path.is_file() else {}
    smoke = bool(selection.get('bootstrap', False))
    sources = {}

    def record_source(path):
        sources[str(path.resolve())] = {'sha256': file_sha256(path), 'bytes': path.stat().st_size}

    if selection_path.is_file():
        record_source(selection_path)
    record_source(Path(__file__))
    manifest = source / 'frozen_manifest.json'
    if manifest.is_file():
        record_source(manifest)
        expected = selection.get('frozen_manifest_sha256')
        if expected and file_sha256(manifest) != expected:
            raise ValueError('Frozen manifest hash differs from the recorded experiment selection')
    runs, deduplications, configs = {}, {}, {}
    for log in sorted(source.glob('*/train.jsonl')):
        if log.parent.name == 'calibration':
            continue
        runs[log.parent.name], deduplications[log.parent.name] = read_training(log)
        record_source(log)
        configuration = log.parent / 'effective_config.json'
        if configuration.is_file():
            record_source(configuration)
            configs[log.parent.name] = json.loads(configuration.read_text(encoding='utf-8'))
    if not runs:
        raise ValueError(f'No actual training logs found under {source}')
    update_counts = {name: rows[-1]['update'] for name, rows in runs.items()}
    comparable = len(set(update_counts.values())) == 1
    actual = selection.get('actual_updates')
    if actual and actual != update_counts:
        raise ValueError(f'Training log update counts {update_counts} differ from selection {actual}')
    plots = [training_figure(runs, selection, output, smoke)]
    for split, key in (('validation', 'validation'), ('test', 'test')):
        if selection.get(key):
            plots.append(metric_figure(selection[key], split, selection, output, smoke))
    budgets = source / 'budget.json'
    if budgets.is_file():
        record_source(budgets)
    provenance = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'input': str(source), 'output': str(output),
        'bootstrap': smoke, 'banner': 'SMOKE - pipeline verification only' if smoke else 'Actual experiment results',
        'sources': sources, 'plots': {str(path): {'sha256': file_sha256(path)} for path in plots},
        'actual_updates': update_counts, 'same_final_update_count': comparable,
        'manifest_sha256_by_run': {name: cfg.get('manifest_sha256') for name, cfg in configs.items()},
        'duplicate_update_rows_last_record_kept': deduplications,
        'selection': {'chosen': selection.get('chosen'), 'fallback_to_single': selection.get('fallback_to_single'),
                      'target_10_percent_achieved': selection.get('target_10_percent_achieved')},
        'plotted_metrics': {key: selection.get(key, {}) for key in ('validation', 'test')},
        'notes': ['Plots contain actual recorded values only; no interpolated data points, invented samples, or statistical uncertainty estimates.',
                  'The smoke sample size cannot establish full course performance.' if smoke else 'Selection and test values follow the supplied experiment record.',
                  'Matching update counts are necessary but do not alone prove identical data, initialization, or cumulative GPU budget.'],
    }
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'plots': [str(path) for path in plots], 'provenance': str(output / 'provenance.json'),
                      'bootstrap': smoke, 'actual_updates': update_counts, 'same_final_update_count': comparable}, indent=2))


if __name__ == '__main__':
    main()
