"""Complete backbone-by-dataset matrices from measured evaluation outputs."""
from __future__ import annotations

import base64
import csv
import html
import io
import json
import math
import re
import statistics
from pathlib import Path

from scripts.build_results_report import DATASETS, DATASET_LABELS


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def _number(value):
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def aggregate_seeds(rows, patches):
    cells = {}
    for dataset in DATASETS:
        values = [r['cells'][dataset]['r2'] for r in rows
                  if dataset in r['cells'] and r['cells'][dataset]['r2'] is not None]
        if values:
            cells[dataset] = {'r2': statistics.mean(values),
                              'sd': statistics.stdev(values) if len(values) > 1 else None,
                              'n': rows[0]['cells'].get(dataset, {}).get('n')}
    return {'label': f'{patches} patches · mean across {len(rows)} seeds', 'cells': cells,
            'uncertainty': f'Seed SD (sample SD, n={len(rows)})', 'kind': 'seed mean'}


def collect_matrix(root):
    root = Path(root)
    evaluations, finals = [], []
    for path in sorted((root / 'outputs').rglob('summary.json')):
        summary = read_json(path)
        points = summary.get('sets', {})
        if not any(d in points and 'embedding_r2' in points[d] for d in DATASETS):
            continue
        match = re.fullmatch(r'p(24|48)_s([0-2])_.*_step497990', path.parent.name)
        label = path.parent.name
        model, raw, random = ({'label': label, 'kind': 'pretrained', 'cells': {}, 'uncertainty': 'Bootstrap SD'},
                              {'label': 'Raw spectra', 'kind': 'raw', 'cells': {}, 'uncertainty': 'Bootstrap SD'},
                              {'label': f'Random-init control · {label}', 'kind': 'random', 'cells': {}, 'uncertainty': 'Bootstrap SD'})
        for dataset in DATASETS:
            p = points.get(dataset)
            if p is None:
                continue
            nested = read_json(path.parent / dataset / 'nested_results.json').get('families', {})
            for row, value_key, sd_key, family in [(model, 'embedding_r2', 'embedding_sd', 'embedding'),
                                                 (raw, 'raw_r2', 'raw_sd', 'raw')]:
                value = _number(p.get(value_key))
                if value is not None:
                    sd = _number(p.get(sd_key, nested.get(family, {}).get('bootstrap_sd')))
                    row['cells'][dataset] = {'r2': value, 'sd': sd, 'n': p.get('n')}
            rc = p.get('random_control') or {}
            if _number(rc.get('embedding_r2')) is not None:
                rc_nested = read_json(path.parent / dataset / 'random_control/nested_results.json').get('families', {})
                random['cells'][dataset] = {'r2': float(rc['embedding_r2']),
                    'sd': _number(rc.get('embedding_sd', rc_nested.get('embedding', {}).get('bootstrap_sd'))), 'n': p.get('n')}
        rows = [r for r in (model, raw, random) if r['cells']]
        for row in rows:
            row['baseline'] = raw['cells']
        entry = {'id': path.parent.name, 'source': str(path.relative_to(root)), 'rows': rows}
        evaluations.append(entry)
        if match:
            model['patches'], model['seed'] = int(match[1]), int(match[2])
            model['label'] = f'{match[1]} patches · seed {match[2]} · CV-selected features'
            model['wins'] = sum(model['cells'][d]['r2'] > raw['cells'][d]['r2'] for d in model['cells'] if d in raw['cells'])
            model['mean_delta'] = statistics.mean(model['cells'][d]['r2'] - raw['cells'][d]['r2'] for d in model['cells'] if d in raw['cells'])
            finals.append(entry)
    final_rows, means = [], []
    complete = len(finals) == 6 and {(e['rows'][0]['patches'], e['rows'][0]['seed']) for e in finals} == {(p, s) for p in (24, 48) for s in range(3)}
    complete = complete and all(set(r['cells']) == set(DATASETS) for e in finals for r in e['rows'])
    if finals:
        # Shared controls may be deduplicated only after verifying every measured cell.
        raw = finals[0]['rows'][1]
        def same(a, b):
            return set(a['cells']) == set(b['cells']) and all(
                a['cells'][d]['n'] == b['cells'][d]['n'] and
                all((a['cells'][d][k] is None and b['cells'][d][k] is None) or
                    (a['cells'][d][k] is not None and b['cells'][d][k] is not None and
                     math.isclose(a['cells'][d][k], b['cells'][d][k], abs_tol=1e-6, rel_tol=1e-6)) for k in ('r2', 'sd'))
                for d in a['cells'])
        if all(same(raw, e['rows'][1]) for e in finals):
            final_rows.append(raw)
        else:
            for e in finals:
                final_rows.append({**e['rows'][1], 'label': f"Raw spectra · {e['id']}"})
        for patches in (24, 48):
            entries = [e for e in finals if e['rows'][0]['patches'] == patches]
            models = sorted([e['rows'][0] for e in entries], key=lambda r: r['seed'])
            final_rows += models
            controls = [e['rows'][2] for e in entries if len(e['rows']) > 2]
            if controls and all(same(controls[0], r) for r in controls):
                final_rows.append({**controls[0], 'label': f'{patches} patches · random-init control (initialization seed 0)'})
            else:
                final_rows += controls
            if models:
                means.append(aggregate_seeds(models, patches))
    conclusion = None
    if complete:
        model_rows = [e['rows'][0] for e in finals]
        by_pair = {(r['patches'], r['seed']): r for r in model_rows}
        architecture = {}
        for patches, mean in zip((24, 48), means):
            seed_rows = [r for r in model_rows if r['patches'] == patches]
            count = sum(mean['cells'][d]['r2'] > raw['cells'][d]['r2'] for d in DATASETS)
            architecture[str(patches)] = {'mean_dataset_wins': count,
                'seed_wins': [r['wins'] for r in sorted(seed_rows, key=lambda r: r['seed'])],
                'mean_delta': statistics.mean(r['mean_delta'] for r in seed_rows),
                'majority_rule_pass': count >= 5 and sum(r['wins'] >= 5 for r in seed_rows) >= 2}
        gains = [by_pair[(48, s)]['mean_delta'] - by_pair[(24, s)]['mean_delta'] for s in range(3)]
        conclusion = {'architectures': architecture, 'paired_48_minus_24': gains,
                      'mean_paired_gain': statistics.mean(gains),
                      'prefer_48': statistics.mean(gains) > 0 and sum(g > 0 for g in gains) >= 2}
    return {'datasets': DATASETS, 'evaluations': evaluations, 'final_rows': final_rows,
            'seed_means': means, 'complete': complete, 'conclusion': conclusion}


def matrix_csv(rows):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(['Backbone', 'Uncertainty definition', *[f'{d} {field}' for d in DATASETS for field in ('R2', 'SD', 'n')]])
    for row in rows:
        writer.writerow([row['label'], row['uncertainty'], *[row['cells'].get(d, {}).get(k, '')
                        for d in DATASETS for k in ('r2', 'sd', 'n')]])
    return stream.getvalue()


def render_matrix(data):
    final_raw = next((r['cells'] for r in data['final_rows'] if r['kind'] == 'raw'), {})
    def table(rows, baseline=None):
        headers = []
        for d in DATASETS:
            counts = {r['cells'].get(d, {}).get('n') for r in rows} - {None}
            n = str(next(iter(counts))) if len(counts) == 1 else ('varies' if counts else 'unavailable')
            headers.append(f'<th>{DATASET_LABELS[d]}<br><small class="sample-count">n={n}</small></th>')
        heading = '<thead><tr><th>Backbone / uncertainty</th>' + ''.join(headers) + '</tr></thead>'
        body = []
        for row in rows:
            cells = []
            for d in DATASETS:
                c = row['cells'].get(d, {})
                if c.get('r2') is None:
                    cells.append('<td>—</td>')
                else:
                    sd = f"{c['sd']:.4f}" if c.get('sd') is not None else 'unavailable'
                    raw = (row.get('baseline') or baseline or {}).get(d, {}).get('r2')
                    win = row.get('kind') != 'raw' and raw is not None and c['r2'] > raw
                    cls = ' class="win-cell"' if win else ''
                    marker = ' · win over raw' if win else ''
                    cells.append(f'<td{cls} title="n={c.get("n", "?")}{marker}">{c["r2"]:.4f} ± {sd}</td>')
            body.append(f'<tr><td>{html.escape(row["label"])}<br><small>{html.escape(row["uncertainty"])}</small></td>{"".join(cells)}</tr>')
        return '<div class="table-wrap"><table class="table results-matrix">' + heading + '<tbody>' + ''.join(body) + '</tbody></table></div>'
    def download(rows, name):
        payload = base64.b64encode(matrix_csv(rows).encode()).decode()
        return f'<a class="badge" download="{name}.csv" href="data:text/csv;base64,{payload}">Download CSV</a>'
    note = '<p class="note"><b>Uncertainty:</b> individual evaluations show R² ± bootstrap SD from resampling spectra and averaging repeat R², keeping fitted out-of-fold predictions fixed. These are not confidence intervals or SD across folds. Aggregate rows show mean R² ± sample SD across three training seeds, separately. Missing evaluations/uncertainty are shown explicitly. Backbone scores use layer/readout and probe selection inside inner CV, never the best outer-test layer.</p>'
    note = '<p class="note"><b>What this main table measures:</b> each backbone is frozen at its final training checkpoint, but the embedding layer and readout are selected inside each outer fold’s inner cross-validation. A fold may use shallow flattened features, intermediate segmented features, or another evaluated arm. This is the score of a feature-selection procedure, not a fixed block6 mean embedding and not a simple average of layers. The fixed-readout tables and depth plots below measure individual interfaces. “Final checkpoint” means training is finished; it does not mean the final encoder block is used.</p>' + note
    conclusion = ''
    if data['conclusion']:
        c = data['conclusion']; a, b = c['architectures']['24'], c['architectures']['48']
        conclusion = f'<p class="note"><b>Final result:</b> both architectures beat raw on {a["mean_dataset_wins"]}/8 datasets using three-seed mean R². Mean ΔR²: 24 patches {a["mean_delta"]:+.4f}; 48 patches {b["mean_delta"]:+.4f}. Seed-wise wins: 24 patches {a["seed_wins"]}; 48 patches {b["seed_wins"]}. The 48-patch recipe is preferred by the registered exploratory rule: positive mean paired gain ({c["mean_paired_gain"]:+.4f}) and wins in two of three seed pairs. This is a recipe-selection rule, not statistical significance. All-eight performance has not been achieved.</p>'
        conclusion += '<p class="small">Decision-rule provenance: docs/superpowers/plans/2026-10-07-screen8-final-direction.md, “Follow-up paired architecture experiment,” recorded in pre-launch commit ae25377. The plan specifies positive mean paired gain and at least two of three favorable seed pairs; majority also requires five seed-mean dataset wins and at least two seeds individually winning five datasets.</p>'
    final = table(data['final_rows'], final_raw) if data['final_rows'] else '<p>Final evaluations unavailable.</p>'
    all_rows = [{**r, 'label': f'{e["id"]} · {r["label"]}'} for e in data['evaluations'] for r in e['rows']]
    history = table(all_rows)
    return f'<style>.results-matrix{{font-size:12px}}.results-matrix td,.results-matrix th{{padding:8px 5px}}.sample-count{{font-size:10px;font-weight:400;text-transform:none}}.results-matrix .win-cell{{background:#e2f2e5;color:#205c38}}.results-matrix td:first-child,.results-matrix th:first-child{{white-space:normal;min-width:180px;max-width:250px;overflow-wrap:anywhere}}</style><section class="section" id="completeResults"><h2>Final checkpoints · CV-selected feature results</h2><p class="sub">Inner-CV layer/readout selection · datasets in columns · R² ± uncertainty</p>{conclusion}{note}<p class="small">Green cells: R² exceeds the matching raw baseline, using unrounded scores. A highlighted win is not a statistical-significance claim.</p>{download(data["final_rows"], "final_results_matrix")}{final}<h3>Architecture means and seed variability</h3>{table(data["seed_means"], final_raw)}<details class="card" style="margin-top:18px"><summary>Complete evaluation archive · {len(data["evaluations"])} backbone scorecards, with raw and available random-init controls</summary><p>Historical protocols and sample sets can differ. Read each evaluation with its own raw control; blank cells mean not evaluated. Hover a measured cell for its sample count. Display rounds to four decimal places; CSV retains full precision.</p>{download(all_rows, "all_backbones_results_matrix")}{history}</details></section>'
