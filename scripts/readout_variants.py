"""Fixed block/readout evidence from existing nested evaluations; no retraining."""
from __future__ import annotations

import html
import json
import re
import statistics
from pathlib import Path

import numpy as np

from scripts.build_results_report import DATASETS, DATASET_LABELS
from scripts.results_matrix import aggregate_seeds, read_json


def collect_variants(root):
    rows, raw_rows, controls = [], [], []
    for path in sorted((Path(root) / 'outputs').rglob('summary.json')):
        match = re.fullmatch(r'p(24|48)_s([0-2])_.*_step497990', path.parent.name)
        if not match:
            continue
        patches, seed = int(match[1]), int(match[2])
        summary = read_json(path)
        baseline = {d: {'r2': p['raw_r2'], 'sd': p.get('raw_sd'), 'n': p.get('n')}
                    for d, p in summary.get('sets', {}).items() if d in DATASETS}
        raw_rows.append({'patches': patches, 'seed': seed, 'cells': baseline, 'kind': 'raw',
                         'label': 'Raw spectra', 'uncertainty': 'Bootstrap SD'})
        for kind, prefix in [('pretrained', path.parent), ('random', path.parent)]:
            arms = {}
            for dataset in DATASETS:
                nested_path = prefix / dataset / ('random_control/nested_results.json' if kind == 'random' else 'nested_results.json')
                for arm, family in read_json(nested_path).get('families', {}).items():
                    parsed = re.fullmatch(r'layer([0-6])(?:/(seg4|flat))?', arm)
                    if not parsed:
                        continue
                    stage, readout = int(parsed[1]), parsed[2] or 'mean'
                    row = arms.setdefault(arm, {'patches': patches, 'seed': seed, 'arm': arm,
                        'stage': stage, 'readout': readout, 'kind': kind, 'baseline': baseline,
                        'dimension': {'mean': 256, 'seg4': 1024, 'flat': patches * 256}[readout],
                        'label': f'{patches}p · seed {seed} · {arm}' + (' · random-init' if kind == 'random' else ''),
                        'uncertainty': 'Bootstrap SD', 'cells': {}, 'source': str(nested_path.relative_to(root))})
                    row['cells'][dataset] = {'r2': family['r2_mean'], 'sd': family.get('bootstrap_sd'),
                                              'n': baseline.get(dataset, {}).get('n')}
                    row.setdefault('sources', {})[dataset] = str(nested_path.relative_to(root))
            (rows if kind == 'pretrained' else controls).extend(arms.values())
    arms = sorted({r['arm'] for r in rows}, key=lambda a: (int(a[5]), a))
    return {'rows': rows, 'raw_rows': raw_rows, 'controls': controls, 'arms': arms}


def depth_figure(data, patches):
    import matplotlib.pyplot as plt
    colors = {'mean': '#3976b8', 'seg4': '#087e83', 'flat': '#d18a25'}
    fig, axes = plt.subplots(2, 4, figsize=(15, 8), squeeze=False)
    for ax, dataset in zip(axes.flat, DATASETS):
        raw = next((r['cells'][dataset]['r2'] for r in data['raw_rows']
                    if r['patches'] == patches and dataset in r['cells']), None)
        if raw is not None:
            ax.axhline(raw, color='#596b74', ls='--', lw=1.3, label='Raw')
        for readout in ('mean', 'seg4', 'flat'):
            stages, means, deviations = [], [], []
            for stage in range(7):
                candidates = [r for r in data['rows'] if r['patches'] == patches and
                              r['readout'] == readout and r['stage'] == stage and dataset in r['cells']]
                if len(candidates) != 3 or {r['seed'] for r in candidates} != {0, 1, 2}:
                    continue
                values = [r['cells'][dataset]['r2'] for r in candidates]
                stages.append(stage); means.append(statistics.mean(values)); deviations.append(statistics.stdev(values))
                ax.scatter([stage] * 3, values, s=9, color=colors[readout], alpha=.35)
            if stages:
                mean, sd = np.array(means), np.array(deviations)
                ax.plot(stages, mean, 'o-', color=colors[readout], ms=4,
                        label=f'{readout} ({256 if readout == "mean" else 1024 if readout == "seg4" else patches * 256}d)')
                ax.fill_between(stages, mean - sd, mean + sd, color=colors[readout], alpha=.12)
        ax.axvspan(5.75, 6.25, color='#d18a25', alpha=.09)
        ax.set_xticks(range(7), ['0\ntokens', '1', '2', '3', '4', '5', '6\nfinal'])
        ax.set(title=DATASET_LABELS[dataset], xlabel='Encoder block', ylabel='Nested-CV R²')
        ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f'Where the signal lives · {patches} patches\nMean R² ± sample SD across three seeds · fixed readout at each block', fontsize=14)
    return fig


def render_variants(data):
    if not data['rows']:
        return ''
    def table(rows):
        header = '<thead><tr><th>Fixed backbone readout</th>' + ''.join(
            f'<th>{DATASET_LABELS[d]}<br><small class="sample-count">n={next((r["cells"][d]["n"] for r in rows if d in r["cells"]), "unavailable")}</small></th>' for d in DATASETS) + '</tr></thead>'
        body = []
        for r in rows:
            cells = []
            for d in DATASETS:
                c = r['cells'].get(d)
                if not c:
                    cells.append('<td>—</td>'); continue
                baseline = r.get('baseline', {}).get(d, {}).get('r2')
                win = r['kind'] != 'raw' and baseline is not None and c['r2'] > baseline
                sd = f'{c["sd"]:.4f}' if c['sd'] is not None else 'unavailable'
                cls = ' class="win-cell"' if win else ''
                cells.append(f'<td{cls}>{c["r2"]:.4f} ± {sd}</td>')
            wins = sum(c['r2'] > r.get('baseline', {}).get(d, {}).get('r2', float('inf')) for d, c in r['cells'].items())
            note = '' if r['kind'] == 'raw' else f' · {wins}/8 wins'
            body.append(f'<tr><td>{html.escape(r["label"])}{note}<br><small>{r["uncertainty"]}</small></td>{"".join(cells)}</tr>')
        return '<div class="table-wrap"><table class="table results-matrix">' + header + '<tbody>' + ''.join(body) + '</tbody></table></div>'
    panels = []
    for arm in data['arms']:
        models = [r for r in data['rows'] if r['arm'] == arm]
        display = [data['raw_rows'][0], *models]
        aggregates = []
        for patches in (24, 48):
            group = [r for r in models if r['patches'] == patches]
            if len(group) == 3:
                mean = aggregate_seeds(group, patches)
                mean['label'] += f' · {arm}'
                mean['baseline'] = group[0]['baseline']
                aggregates.append(mean)
            rc = next((r for r in data['controls'] if r['patches'] == patches and r['arm'] == arm), None)
            if rc:
                display.append({**rc, 'label': f'{patches}p · {arm} · random-init (seed 0)'})
        key = arm.replace('/', '_')
        hidden = '' if arm == 'layer2/flat' or 'layer2/flat' not in data['arms'] and arm == data['arms'][0] else ' hidden'
        dimensions = ' / '.join(f'{patches}p: {next(r["dimension"] for r in models if r["patches"] == patches):,}d'
                                for patches in (24, 48) if any(r['patches'] == patches for r in models))
        panels.append(f'<div class="variant-panel" data-arm="{key}"{hidden}><h3>{arm} · {dimensions}</h3>{table(display)}<h4>Three-seed means ± seed SD</h4>{table(aggregates)}</div>')
    options = ''.join(f'<option value="{a.replace("/", "_")}"{" selected" if a == "layer2/flat" else ""}>{a}</option>' for a in data['arms'])
    return f'''<section class="section" id="readoutVariants"><h2>Fixed block and readout variants</h2><p class="sub">Same trained checkpoints, different frozen feature interfaces; no best-layer selection within each row.</p><p class="note">These are measured, fixed-family nested-CV scores: only normalization and probe are selected inside the folds. Inspecting all layers now is retrospective development analysis, not independent confirmation of the chosen production readout. Green cells exceed the matching raw baseline. Individual rows use bootstrap SD; aggregate rows use sample SD across three seeds.</p><div class="controls"><label for="variantSelect">FIXED READOUT</label><select id="variantSelect">{options}</select></div>{''.join(panels)}<p class="small">Layer0 is the tokenizer; layer1–5 are intermediate block states; layer6 is the final six-block output after LayerNorm. Mean:256 dimensions; seg4:1,024; flat:6,144 (24 patches) or12,288 (48 patches). Flattened features were evaluated only at layers0–2; absent later measurements are not extrapolated.</p></section><script>(()=>{{const s=document.getElementById('variantSelect');s.addEventListener('change',()=>{{document.querySelectorAll('.variant-panel').forEach(p=>{{p.hidden=p.dataset.arm!==s.value}})}})}})();</script>'''
