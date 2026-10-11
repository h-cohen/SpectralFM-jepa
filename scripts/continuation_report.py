"""Offline status and paired continuation results."""
import html
import json
import re
from pathlib import Path

import numpy as np

from scripts.build_results_report import DATASETS, DATASET_LABELS


def assess_study(data, baseline):
    result = {}
    jobs = {(j['seed'], j['arm']): j.get('results') for j in data['jobs'].values()}
    for policy in ('fixed_block2', 'menu'):
        if any(not jobs.get((s, a)) for s in (202, 203) for a in ('control', 'low_sigreg')):
            result[policy] = 'pending'; continue
        base = baseline['scorecard'][policy]
        fixed_floor = baseline['scorecard']['fixed_block2']['mean_delta'] - .005
        def preserves(seed, arm):
            score = jobs[seed, arm]['scorecard']
            return score['fixed_block2']['mean_delta'] >= fixed_floor and score[policy]['wins'] >= base['wins']
        duration, lower = [], []
        for seed in (202, 203):
            c = jobs[seed, 'control']['scorecard'][policy]
            l = jobs[seed, 'low_sigreg']['scorecard'][policy]
            duration.append(preserves(seed, 'control') and c['mean_delta'] > base['mean_delta'])
            lower.append(preserves(seed, 'low_sigreg') and l['wins'] >= c['wins']
                         and l['mean_delta'] > max(base['mean_delta'], c['mean_delta']))
        result[policy] = 'lower_sigreg' if all(lower) else 'duration_only' if all(duration) else 'inconclusive'
    return result


def collect_study(root):
    base = Path(root) / 'outputs/continuation-study'
    if not (base / 'manifest.json').exists():
        return {}
    data = json.loads((base / 'manifest.json').read_text())
    for name, job in data['jobs'].items():
        for key, file in [('progress', 'state.json'), ('results', 'policies/policy_summary.json')]:
            path = base / name / file
            if path.exists():
                job[key] = json.loads(path.read_text())
        log = base / name / 'pipeline.log'
        if log.exists():
            match = re.search(r'https://wandb.ai/hcohen/spectralfm-lejepa/runs/(\w+)', log.read_text())
            if match:
                job['run_id'] = match.group(1)
    baseline = Path(root) / data['baseline_policies']
    if baseline.exists():
        scores = json.loads(baseline.read_text())
        data['baseline_scores'] = scores['scorecard']
        data['preference_result'] = assess_study(data, scores)
    return data


def render_study(data):
    if not data:
        return ''
    body = ['<section class="section" id="continuationStudy"><h2>Matched continuation · duration versus objective balance</h2><p>Two paired continuation seeds, each comparing unchanged SIGReg λ=.05 with λ=.01. Same seed101 starting checkpoint; fresh AdamW; 99,598 additional steps, 1,000 warmup steps, peak LR 5e-5, floor 5e-7. The frozen completed model remains the no-training baseline.</p><p class="note">Exploratory benchmark research. Two continuation seeds do not replace independent pretraining replication or deployment validation. Original export and failed qualification remain unchanged.</p><table class="table"><thead><tr><th>Run</th><th>λ</th><th>State</th><th>GPU (logical)</th></tr></thead><tbody>']
    for name, job in data['jobs'].items():
        state = job.get('progress', {}).get('state', 'registered')
        label = html.escape(name)
        if job.get('run_id'):
            label = f'<a href="https://wandb.ai/hcohen/spectralfm-lejepa/runs/{job["run_id"]}">{label}</a>'
        body.append(f'<tr><td>{label}</td><td>{job["lambda_sigreg"]}</td><td>{html.escape(state)}</td><td>{job["logical_gpu"]}</td></tr>')
    body.append('</tbody></table><p class="small">W&B group: <a href="https://wandb.ai/hcohen/spectralfm-lejepa/groups/continuation-study">continuation-study</a>. Training records losses, gradient norm, learning rate, progress, validation and geometry; final audits measure per-term block2/block6 gradients. Final probes use fixed block2 and the predeclared four-readout menu. Other blocks are diagnostic; intermediate checkpoints do not select winners.</p>')
    finished = {n: j for n, j in data['jobs'].items() if j.get('results')}
    if finished:
        body.append('<div class="table-wrap"><table class="table results-matrix"><thead><tr><th>Final interface · R² ± bootstrap SD</th>' + ''.join(f'<th>{DATASET_LABELS[d]}<br><small class="sample-count">n={next(iter(finished.values()))["results"]["sets"][d]["n"]}</small></th>' for d in DATASETS) + '</tr></thead><tbody>')
        for name, job in finished.items():
            for policy in ('fixed_block2', 'menu'):
                for random in (False, True):
                    cells = []
                    for ds in DATASETS:
                        p = job['results']['sets'][ds]['policies'][policy]
                        raw = job['results']['sets'][ds]['policies']['raw']['r2']
                        value, sd = (p['random_r2'], p['random_sd']) if random else (p['r2'], p['sd'])
                        cls = ' class="win-cell"' if value > raw else ''
                        cells.append(f'<td{cls}>{value:.4f} ± {sd:.4f}</td>')
                    body.append(f'<tr><td>{name} · {policy}{" · random" if random else ""}</td>{"".join(cells)}</tr>')
        body.append('</tbody></table></div>')
        for policy in ('fixed_block2', 'menu'):
            for arm in ('control', 'low_sigreg'):
                scores = [j['results']['scorecard'][policy]['mean_delta'] for j in finished.values() if j['arm'] == arm]
                if len(scores) == 2:
                    body.append(f'<p>{arm} · {policy}: mean raw gain {np.mean(scores):+.4f} ± {np.std(scores, ddof=1):.4f} across two continuation seeds (sample SD; not bootstrap SD).</p>')
    else:
        body.append('<p>Final comparisons pending. The existing 7/8 menu result belongs to the frozen baseline, not these new runs.</p>')
    if data.get('preference_result'):
        body.append('<p>Research preference: ' + html.escape(str(data['preference_result'])) + '. This is a predeclared development rule, not a release qualification.</p>')
        if data.get('baseline_scores') and finished:
            baseline = data['baseline_scores']
            body.append('<table class="table"><thead><tr><th>Run / interface</th><th>Raw wins</th><th>Mean raw gain</th><th>Change from frozen baseline</th></tr></thead><tbody>')
            for name, job in finished.items():
                for policy in ('fixed_block2', 'menu'):
                    s = job['results']['scorecard'][policy]
                    body.append(f'<tr><td>{name} / {policy}</td><td>{s["wins"]}/8</td><td>{s["mean_delta"]:+.4f}</td><td>{s["mean_delta"]-baseline[policy]["mean_delta"]:+.4f}</td></tr>')
            body.append('</tbody></table>')
    body.append('<p>Predeclared research preference: lower λ must improve mean raw gain versus frozen baseline and matched control on both seeds without reducing raw win count; block2 mean R² must stay within .005 of baseline. Inspect task regressions and uncertainty. If this fails, retain the current model and report the experiment as inconclusive. Production selection still requires independent labelled development and a reserved release holdout.</p></section>')
    return ''.join(body)
