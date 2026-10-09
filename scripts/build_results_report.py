"""Build a self-contained, interactive HTML report from local evaluation scorecards."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DATASETS = [
    'labeled_data', 'dataset0055', 'dataset0106', 'dataset0109',
    'dataset0112', 'dataset0113', 'dataset0114', 'dataset0120',
]
DATASET_LABELS = {
    'labeled_data': 'Labeled data',
    'dataset0055': '0055', 'dataset0106': '0106', 'dataset0109': '0109',
    'dataset0112': '0112', 'dataset0113': '0113', 'dataset0114': '0114',
    'dataset0120': '0120',
}


def _normalized_set(point):
    random = point.get('random_control') or {}
    paired_raw = point.get('embedding_minus_raw') or {}
    paired_random = point.get('vs_random_control') or {}
    model_r2 = point.get('embedding_r2', point.get('model_r2'))
    raw_r2 = point.get('raw_r2')
    random_r2 = random.get('embedding_r2', point.get('random_r2'))
    delta_raw = point.get('delta_vs_raw', paired_raw.get('delta'))
    if delta_raw is None and _finite(model_r2) and _finite(raw_r2):
        delta_raw = model_r2 - raw_r2
    delta_random = paired_random.get('delta')
    if delta_random is None and _finite(model_r2) and _finite(random_r2):
        delta_random = model_r2 - random_r2
    return {
        'n': point.get('n'),
        'raw_r2': raw_r2,
        'model_r2': model_r2,
        'random_r2': random_r2,
        'delta_vs_raw': delta_raw,
        'sd_vs_raw': paired_raw.get('sd'),
        'delta_vs_random': delta_random,
        'sd_vs_random': paired_random.get('sd'),
    }


def _classify(path):
    name = path.parent.name
    step_match = re.search(r'_step(\d+)$', name)
    if not step_match:
        return None
    seed_match = re.search(r'_s([0-2])(?:_|$)', name)
    if name.startswith(('p48_lr50_', 'p48_s')):
        family, patches = '48-patch', 48
        # The screen-7 arm used the configured default seed 0 and omitted it in its name.
        seed = int(seed_match.group(1)) if seed_match else 0
    elif name.startswith(('lr50_s', 'p24_s')):
        family, patches = '24-patch', 24
        if not seed_match:
            return None
        seed = int(seed_match.group(1))
    else:
        return None
    return family, patches, seed, int(step_match.group(1))


def collect_report_data(repo_root):
    """Load comparable checkpoints, preserving absent controls as null."""
    root = Path(repo_root)
    records = []
    for path in (root / 'outputs').glob('*/eval/*/summary.json'):
        classified = _classify(path)
        if classified is None:
            continue
        family, patches, seed, step = classified
        try:
            summary = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        sets = summary.get('sets', {})
        if not all(dataset in sets for dataset in DATASETS):
            continue
        score = summary.get('scorecard', {}).get('model', {})
        records.append({
            'id': path.parent.name,
            'cohort': 'fresh paired study' if path.parent.name.startswith(('p24_s', 'p48_s')) else 'historical',
            'label': f'{patches} patches · seed {seed} · {step:,}',
            'family': family,
            'patches': patches,
            'seed': seed,
            'step': step,
            'wins': score.get('wins'),
            'strong_wins': score.get('strong_wins'),
            'mean_delta': score.get('mean_delta'),
            'status': 'exploratory single seed' if family == '48-patch' else 'measured checkpoint',
            'sets': {dataset: _normalized_set(sets[dataset]) for dataset in DATASETS},
        })
    full_48_seeds = {row['seed'] for row in records
                     if row['family'] == '48-patch' and row['step'] >= 497990}
    for row in records:
        if row['family'] == '48-patch':
            row['status'] = ('confirmatory final checkpoint' if row['step'] >= 497990 and len(full_48_seeds) >= 3
                             else 'exploratory single seed')
    records.sort(key=lambda row: (row['patches'], row['seed'], row['step']))

    decision_path = root / 'outputs/screen-8/decision.json'
    decision = json.loads(decision_path.read_text()) if decision_path.exists() else None
    study_path = root / 'outputs/paired-long/manifest.json'
    study = json.loads(study_path.read_text()) if study_path.exists() else None
    story_path = root / 'outputs/training-progress/story_evidence.json'
    story = json.loads(story_path.read_text()) if story_path.exists() else None
    sensitivity_path = root / 'outputs/screen-1/results.json'
    sensitivity = json.loads(sensitivity_path.read_text()) if sensitivity_path.exists() else []
    sensitivity = [r for r in sensitivity if r.get('arm') in ('control_s0', 'control_s1', 'lambda02')]
    from scripts.local_wandb import paired_histories, timestamp_text
    histories, history_errors = paired_histories(root, study)
    from scripts.results_matrix import collect_matrix
    matrix = collect_matrix(root)
    from scripts.readout_variants import collect_variants
    variants = collect_variants(root)
    candidate_path = root / 'outputs/final-model/manifest.json'
    candidate = json.loads(candidate_path.read_text()) if candidate_path.exists() else None
    if candidate and candidate.get('run_id'):
        observed, errors = paired_histories(root, {'runs': [candidate]})
        if observed and observed[0].get('latest_step') is not None:
            candidate['step'] = observed[0]['latest_step']
            if observed[0].get('latest_timestamp'):
                candidate['snapshot_at'] = timestamp_text(observed[0]['latest_timestamp'])
        candidate['history_errors'] = errors
        assessment = root / 'outputs/final-model/assessment.json'
        if assessment.exists():
            candidate['assessment'] = json.loads(assessment.read_text())
    if histories and study:
        by_run = {r['run_id']: r for r in histories}
        for run in study['runs']:
            if run.get('run_id') in by_run:
                source = by_run[run['run_id']]
                run['step'] = source['latest_step']
                run['state'] = 'latest local W&B observation; live service state unverified'
        timestamps = [r['latest_timestamp'] for r in histories if r.get('latest_timestamp')]
        if timestamps:
            study['snapshot_at'] = timestamp_text(max(timestamps)) + ' (local histories)'
    if matrix['complete'] and study:
        for run in study['runs']:
            run['step'] = 497990
            run['state'] = 'training and eight-dataset evaluation completed'
    return {
        'title': 'SpectralFM training results',
        'generated_from': 'Local nested-CV scorecards; latest long-run checkpoints are final step 497,990.',
        'datasets': [{'id': d, 'label': DATASET_LABELS[d]} for d in DATASETS],
        'records': records,
        'screen8': decision,
        'study': study,
        'story': story,
        'sigreg_sensitivity': sensitivity,
        'local_histories': histories,
        'history_errors': history_errors,
        'matrix': matrix,
        'variants': variants,
        'final_candidate': candidate,
        'evaluation': {
            'protocol': 'Nested cross-validation: 2 outer repeats × 5 folds, with 5-fold inner selection.',
            'win_rule': 'A dataset is a win when pretrained model R² − raw R² > 0; strong win is ≥ +0.05.',
            'note': 'Paired bootstrap SD describes within-scorecard fold uncertainty. Seed-to-seed variation is shown separately; these small-set results are exploratory, not a significance claim.',
        },
    }


def render_html(data, export_dir=None):
    """Render one offline-capable HTML document with inline CSS, data and SVG charts."""
    embedded = json.dumps({k: v for k, v in data.items() if k != 'story'}, ensure_ascii=False, separators=(',', ':'))
    embedded = embedded.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    from scripts.report_story import render_story
    story_html = render_story(data, export_dir=export_dir)
    from scripts.results_matrix import render_matrix
    matrix_html = render_matrix(data['matrix']) if data.get('matrix') else ''
    from scripts.readout_variants import render_variants
    variants_html = render_variants(data['variants']) if data.get('variants') else ''
    from scripts.architecture_diagram import render_dataflow
    flow_html = '<section class="section" id="tensorDataflow"><h2>Architecture and tensor dataflow</h2><p class="sub">Full six-block training objective; fixed two-block production inference. B=256 during training and B=1 for a single spectrum.</p><div class="card">' + render_dataflow() + '</div></section>'
    candidate = data.get('final_candidate')
    candidate_html = ''
    if candidate:
        import html
        run_id = str(candidate.get('run_id', ''))
        link = f'<a href="https://wandb.ai/hcohen/spectralfm-lejepa/runs/{run_id}">W&amp;B training run</a>' if re.fullmatch('[a-z0-9]{8}', run_id) else ''
        candidate_html = f'<section class="section"><h2>Final fixed-interface candidate</h2><div class="card"><p>48 patches · seed101 · fixed block2-flat · 12,288 features</p><p>State: {html.escape(candidate.get("state", "unavailable"))} · step {candidate.get("step", 0):,} / 497,990 · {link}</p><p class="small">Snapshot: {html.escape(candidate.get("snapshot_at", candidate.get("registered_at", "unavailable")))}</p><p>Ten-epoch schedule; no checkpoint or layer search. Automatic final export and fixed-readout evaluation are followed by a predeclared benchmark gate. Passing requires at least5/8 raw wins, positive mean gains versus raw and matched random initialization, and complete finite evaluations with all canaries passing. Deployment-domain validation remains separate.</p></div></section>'
        assessment = candidate.get('assessment')
        if assessment:
            raw_gain = f'{assessment["mean_delta_vs_raw"]:+.4f}' if _finite(assessment.get('mean_delta_vs_raw')) else 'unavailable'
            random_gain = f'{assessment["mean_delta_vs_random"]:+.4f}' if _finite(assessment.get('mean_delta_vs_random')) else 'unavailable'
            result = f'<p class="note">Fixed-readout benchmark: {assessment["wins"]}/8 wins; mean ΔR² versus raw {raw_gain}, versus random {random_gain}. Benchmark qualified: {assessment["benchmark_qualified"]}. Deployment-domain release validation remains separate.</p>'
            candidate_html = candidate_html.replace('</div></section>', result + '</div></section>')
        completed = next((e for e in data.get('matrix', {}).get('evaluations', [])
                          if e['id'].startswith('production_p48_s101_') and e['id'].endswith('_step497990')), None)
        if completed:
            head = '<tr><th>Fixed block2-flat / control</th>' + ''.join(f'<th>{DATASET_LABELS[d]}<br><small class="sample-count">n={completed["rows"][0]["cells"].get(d, {}).get("n", "unavailable")}</small></th>' for d in DATASETS) + '</tr>'
            body = []
            for row in completed['rows']:
                cells = []
                for d in DATASETS:
                    c = row['cells'].get(d)
                    if c is None:
                        cells.append('<td>—</td>'); continue
                    raw = row.get('baseline', {}).get(d, {}).get('r2')
                    win = row['kind'] != 'raw' and raw is not None and c['r2'] > raw
                    cls = ' class="win-cell"' if win else ''
                    sd = f'{c["sd"]:.4f}' if c['sd'] is not None else 'unavailable'
                    cells.append(f'<td{cls}>{c["r2"]:.4f} ± {sd}</td>')
                name = {'pretrained': 'Seed101 · fixed block2-flat', 'raw': 'Raw spectra', 'random': 'Matched random-init · block2-flat'}[row['kind']]
                body.append(f'<tr><td>{name}<br><small>Bootstrap SD</small></td>{"".join(cells)}</tr>')
            score_table = '<div class="table-wrap"><table class="table results-matrix"><thead>' + head + '</thead><tbody>' + ''.join(body) + '</tbody></table></div>'
            candidate_html = candidate_html.replace('</div></section>', score_table + '</div></section>')
    rendered = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SpectralFM | Training results</title>
<style>
:root{--ink:#14212b;--muted:#637581;--line:#dce5e8;--paper:#f4f7f6;--card:#fff;--teal:#087e83;--blue:#3976b8;--gold:#d18a25;--red:#ba5551;--green:#368566;--navy:#102a38;--shadow:0 12px 35px #17374510}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}header{background:radial-gradient(900px 500px at 90% -20%,#216779 0,transparent 65%),linear-gradient(120deg,#102a38,#183c4b);color:#fff;padding:46px max(24px,calc((100vw - 1180px)/2)) 42px}header .eyebrow{color:#91d4d1;text-transform:uppercase;letter-spacing:.15em;font-size:12px;font-weight:750}h1{font-size:clamp(32px,5vw,54px);letter-spacing:-.045em;line-height:1.05;margin:12px 0}header p{max-width:780px;color:#c7d7da;font-size:17px;margin:12px 0 0}.wrap{max-width:1180px;padding:28px 22px 72px;margin:auto}.section{margin:30px 0}.section-head{display:flex;align-items:end;justify-content:space-between;gap:18px;margin-bottom:14px}.section h2{font-size:22px;letter-spacing:-.025em;margin:0}.sub{color:var(--muted);margin:4px 0 0;font-size:13px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.card{background:var(--card);border:1px solid var(--line);border-radius:15px;padding:18px;box-shadow:var(--shadow)}.metric .label{font-size:12px;color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:.08em}.metric strong{display:block;font-size:30px;letter-spacing:-.04em;margin:7px 0 1px}.metric small{color:var(--muted)}.good{color:var(--green)}.warn{color:#a15c18}.bad{color:var(--red)}.chart-card{padding:20px}.controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:12px 0 6px}.controls label{font-size:12px;color:var(--muted);font-weight:700}.controls select{border:1px solid var(--line);border-radius:9px;padding:9px 34px 9px 11px;color:var(--ink);background:white;font:inherit}.chart{width:100%;height:auto;min-height:280px;display:block;overflow:visible}.legend{display:flex;gap:18px;flex-wrap:wrap;color:var(--muted);font-size:12px}.key{display:inline-flex;align-items:center;gap:7px}.dot{width:9px;height:9px;border-radius:50%;display:inline-block}.twocol{display:grid;grid-template-columns:1fr 1fr;gap:14px}.note{border-left:3px solid var(--teal);background:#e7f2f0;padding:14px 16px;border-radius:0 10px 10px 0;color:#31545b}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:12px;background:#fff}.table{border-collapse:collapse;width:100%;font-size:13px}.table th,.table td{text-align:right;padding:10px 12px;border-bottom:1px solid #edf1f2;white-space:nowrap}.table th:first-child,.table td:first-child{text-align:left}.table th{background:#f7f9f9;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.07em;position:sticky;top:0}.table tr:last-child td{border-bottom:0}.pill{display:inline-block;border-radius:99px;padding:3px 8px;font-size:11px;font-weight:750;background:#edf2f2;color:#42616a}.pill.no{background:#faeceb;color:#9b4744}.pill.yes{background:#e8f4ee;color:#27704f}.gate-list{display:grid;grid-template-columns:1fr 1fr;gap:8px}.gate{border:1px solid var(--line);border-radius:10px;padding:11px 12px;display:flex;justify-content:space-between;gap:10px}.gate b{font-size:13px}.gate span{font-size:12px;font-weight:750}.foot{color:var(--muted);font-size:12px;margin-top:16px}.badge{display:inline-flex;align-items:center;gap:8px;border:1px solid #d5e5e1;background:#edf7f3;color:#23644d;border-radius:99px;padding:7px 11px;font-size:12px;font-weight:750}details summary{cursor:pointer;font-weight:700}.small{font-size:12px;color:var(--muted)}
@media(max-width:850px){.grid{grid-template-columns:repeat(2,1fr)}.twocol{grid-template-columns:1fr}.section-head{align-items:start;flex-direction:column}}@media(max-width:520px){.grid{grid-template-columns:1fr 1fr;gap:8px}.metric strong{font-size:24px}.card{padding:14px}.gate-list{grid-template-columns:1fr}header{padding-top:34px}}
@media print{body{background:#fff}.card{box-shadow:none;break-inside:avoid}header{print-color-adjust:exact;-webkit-print-color-adjust:exact}.controls{display:none}.wrap{max-width:none;padding:15px}.section{margin:20px 0}}
</style></head><body>
<header><div class="eyebrow">Spectral foundation models · a measured training story</div><h1>From spectral prediction<br>to transferable features</h1><p>Masked prediction learns spectral structure. SIGReg encourages a balanced feature distribution during training. Follow the visual intuition and measured diagnostics through to held-out prediction across eight datasets.</p></header>
<main class="wrap">
__MATRIX__
__CANDIDATE__
__FLOW__
__VARIANTS__
__STORY__
<section class="section"><div class="section-head"><div><h2>Current result</h2><p class="sub">Final checkpoints · 24 patches · same ten-epoch recipe · raw and random-init controls</p></div><span class="badge">All 12.76M packed spectra eligible · labels excluded from pretraining</span></div><div class="grid" id="metrics"></div></section>
<section class="section"><div class="section-head"><div><h2>Historical checkpoint progression</h2><p class="sub">Historical continuation checkpoints only; fresh paired final results are in the complete table above.</p></div></div><div class="card chart-card"><div class="legend"><span class="key"><i class="dot" style="background:#087e83"></i>Seed 0</span><span class="key"><i class="dot" style="background:#3976b8"></i>Seed 1</span><span class="key"><i class="dot" style="background:#9a5a9b"></i>Seed 2</span><span class="key">Solid: 24 patches · dashed: 48 patches</span></div><svg class="chart" id="timeline" viewBox="0 0 960 330" role="img" aria-label="Dataset wins by checkpoint"></svg><p class="small">A majority requires at least five of eight wins. The existing 48-patch point is one seed at three epochs and is not a long-run comparison.</p></div></section>
<section class="section" id="activeStudySection"><div class="section-head"><div><h2>Confirmatory comparison in progress</h2><p class="sub" id="studySnapshot"></p></div><span class="pill">Six fresh runs · three paired seeds · 10 epochs</span></div><div class="grid" id="activeRuns"></div><p class="foot">Each run logs training metrics directly to W&amp;B. Final eight-dataset evaluations are serialized on the reserved evaluation GPU.</p></section>
<section class="section twocol"><div class="card chart-card"><div class="section-head"><div><h2>Per-dataset view</h2><p class="sub">Inspect raw, pretrained FM, and random-init R² for one measured checkpoint.</p></div></div><div class="controls"><label for="recordSelect">CHECKPOINT</label><select id="recordSelect"></select></div><svg class="chart" id="r2chart" viewBox="0 0 620 390" role="img" aria-label="R squared by dataset"></svg><div class="legend"><span class="key"><i class="dot" style="background:#596b74"></i>Raw</span><span class="key"><i class="dot" style="background:#087e83"></i>Pretrained FM</span><span class="key"><i class="dot" style="background:#d18a25"></i>Random init</span></div></div>
<div class="card chart-card"><div class="section-head"><div><h2>Gain over raw</h2><p class="sub">Positive values are wins; +0.05 marks a strong win. Whiskers show paired bootstrap SD where available.</p></div></div><div class="controls"><label for="deltaSelect">CHECKPOINT</label><select id="deltaSelect"></select></div><svg class="chart" id="deltachart" viewBox="0 0 620 390" role="img" aria-label="Paired R squared gain by dataset"></svg><div class="legend"><span class="key"><i class="dot" style="background:#087e83"></i>FM − raw</span><span class="key"><i class="dot" style="background:#d18a25"></i>FM − random init</span></div></div></section>
<section class="section"><div class="section-head"><div><h2>Screen 8: masking decision</h2><p class="sub">Short matched continuations tested random50 and block75 against random75, across two seeds.</p></div><span class="pill no">No candidate promoted</span></div><div class="card" id="screenDecision"></div></section>
<section class="section"><div class="section-head"><div><h2>Checkpoint scorecards</h2><p class="sub">All displayed values come from completed eight-dataset evaluations. Missing measurements stay blank.</p></div></div><div class="table-wrap"><table class="table" id="scoreTable"></table></div></section>
<section class="section"><div class="note"><b>How to read this.</b> A win is ΔR² &gt; 0 against raw; ≥ +0.05 is a strong win. The protocol uses nested cross-validation (2×5 outer, 5-fold inner). Paired bootstrap SD resamples spectra with fixed out-of-fold predictions; seed variation is the more important uncertainty for recipe selection. The small labeled sets can be noisy. One 48-patch seed and two 24-patch seeds do not establish which architecture is better.</div><details class="card" style="margin-top:14px"><summary>Next experiment and decision rule</summary><p>Run fresh 24- and 48-patch recipes from scratch on the same three seeds, keeping the ten-epoch schedule, data mix, optimizer, masking ratio, normalization, evaluation protocol, and random-init control fixed. Pre-register the final checkpoint for recipe choice; keep intermediate checkpoints descriptive. Compare per-dataset ΔR² by seed and seed means/SD. Promote a recipe only when majority wins repeat across seeds and it improves the cross-seed dataset profile. All-eight remains a separate aspiration, not a claim.</p><p class="small">48-patch evidence currently shown: one seed at 149,397 steps (three epochs), with 5/8 wins. It is exploratory only.</p></details></section>
<p class="foot">Generated from local scorecard summaries. This document embeds its data, styles, and charts and does not load external scripts or fonts. Evaluation scores measure downstream probes; they do not by themselves establish statistical significance.</p>
</main>
<script id="report-data" type="application/json">__DATA__</script>
<script>
(()=>{'use strict';const D=JSON.parse(document.getElementById('report-data').textContent),$=id=>document.getElementById(id),esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),fmt=(v,n=3)=>v==null||!Number.isFinite(v)?'—':Number(v).toFixed(n),pctStep=n=>(n/1000).toFixed(0)+'k',sets=D.datasets,records=D.records;
const latest=(family,seed)=>records.filter(r=>r.family===family&&r.seed===seed).sort((a,b)=>b.step-a.step||Number(b.id.startsWith('p24_s'))-Number(a.id.startsWith('p24_s')))[0];
const f0=latest('24-patch',0),f1=latest('24-patch',1),p48=records.filter(r=>r.family==='48-patch').sort((a,b)=>b.step-a.step)[0];
const avg=(a,b)=>a==null||b==null?null:(a+b)/2;
function metric(label,value,note,cls=''){return `<article class="card metric"><div class="label">${label}</div><strong class="${cls}">${value}</strong><small>${note}</small></article>`}
const allWins=[f0?.wins,f1?.wins].filter(Number.isFinite);$('metrics').innerHTML=[metric('Final seed 0',`${f0?.wins??'—'}/8`,`mean ΔR² ${fmt(f0?.mean_delta)}`,(f0?.wins||0)>=5?'good':'warn'),metric('Final seed 1',`${f1?.wins??'—'}/8`,`mean ΔR² ${fmt(f1?.mean_delta)}`,(f1?.wins||0)>=5?'good':'warn'),metric('Across-seed mean wins',allWins.length?fmt(allWins.reduce((a,b)=>a+b,0)/allWins.length,1)+'/8':'—','Two final seeds; not three-seed confirmation','warn'),metric('48-patch evidence',p48?`${p48.wins}/8`:'—',p48?`${records.filter(r=>r.family==='48-patch').length} scorecard(s); latest ${pctStep(p48.step)} steps`:'No completed scorecard','')].join('');
function svgLine(x1,y1,x2,y2,stroke='#dce5e8',width=1,dash=''){return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${stroke}" stroke-width="${width}" ${dash?`stroke-dasharray="${dash}"`:''}/>`}
function timeline(){if(!$('timeline'))return;const s=$('timeline'),W=960,H=330,L=70,R=24,T=20,B=55,points=records.filter(r=>r.cohort!=='fresh paired study'&&(r.family==='24-patch'||r.family==='48-patch')),max=Math.max(497990,...points.map(x=>x.step)),x=v=>L+v/max*(W-L-R),y=v=>T+(8-v)/8*(H-T-B),colors={0:'#087e83',1:'#3976b8',2:'#9a5a9b'};let out='';for(let w=0;w<=8;w++){out+=svgLine(L,y(w),W-R,y(w),w===5?'#d18a25':'#e6ecee',w===5?1.5:1,w===5?'6 5':'');out+=`<text x="${L-14}" y="${y(w)+4}" text-anchor="end" fill="#637581" font-size="12">${w}</text>`}out+=`<text x="${W-R}" y="${y(5)+14}" text-anchor="end" fill="#a15c18" font-size="11">majority threshold · 5/8</text>`;for(const seed of [0,1,2]){const color=colors[seed];for(const family of ['24-patch','48-patch']){const a=points.filter(r=>r.family===family&&r.seed===seed).sort((p,q)=>p.step-q.step);if(!a.length)continue;const dash=family==='48-patch'?'7 5':'';out+=`<polyline fill="none" stroke="${color}" stroke-width="${family==='24-patch'?3:2.5}" ${dash?`stroke-dasharray="${dash}"`:''} points="${a.map(r=>`${x(r.step)},${y(r.wins)}`).join(' ')}"/>`;for(const r of a){const marker=family==='48-patch'?`<path d="M ${x(r.step)} ${y(r.wins)-7} l 7 13 h -14 z" fill="${color}"/>`:`<circle cx="${x(r.step)}" cy="${y(r.wins)}" r="5" fill="${color}"/>`;out+=`<g>${marker}<title>${family} seed ${seed}, ${r.step.toLocaleString()} steps: ${r.wins}/8 wins</title></g>`}}}for(const v of [0,100000,200000,300000,400000,500000])out+=`<text x="${x(v)}" y="${H-22}" text-anchor="middle" fill="#637581" font-size="11">${v===0?'0':v/1000+'k'}</text>`;s.innerHTML=out}
function options(select){$(select).innerHTML=records.map((r,i)=>`<option value="${i}">${esc(r.label)}${r.status==='exploratory single seed'?' · exploratory':''}</option>`).join('')}
function selected(id){return records[Number($(id).value)||0]}
function r2chart(){const r=selected('recordSelect'),s=$('r2chart'),W=620,H=390,L=74,R=26,T=20,B=70,vals=[];for(const d of sets){const p=r?.sets[d.id]||{};for(const k of ['raw_r2','model_r2','random_r2'])if(Number.isFinite(p[k]))vals.push(p[k])}let min=Math.min(-.1,...vals),max=Math.max(.2,...vals);min=Math.floor(min*10)/10;max=Math.ceil(max*10)/10;const x=v=>L+(v-min)/(max-min)*(W-L-R),y=i=>T+i*(H-T-B)/sets.length;let o='';for(let v=Math.ceil(min*5)/5;v<=max;v+=.2){o+=svgLine(x(v),T,x(v),H-B,'#e6ecee');o+=`<text x="${x(v)}" y="${H-B+18}" text-anchor="middle" fill="#637581" font-size="10">${v.toFixed(1)}</text>`}const cols=['#596b74','#087e83','#d18a25'],keys=['raw_r2','model_r2','random_r2'];sets.forEach((d,i)=>{const p=r?.sets[d.id]||{},yy=y(i);o+=`<text x="${L-9}" y="${yy+4}" text-anchor="end" fill="#42545d" font-size="11">${esc(d.label)}</text>`;keys.forEach((k,j)=>{const v=p[k];if(Number.isFinite(v)){const xx=x(v);o+=`<circle cx="${xx}" cy="${yy+(j-1)*12}" r="4.5" fill="${cols[j]}"><title>${esc(d.label)} · ${['raw','FM','random init'][j]} R² ${fmt(v,4)} · n=${p.n??'?'}</title></circle>`}})});o+=`<text x="${(L+W-R)/2}" y="${H-22}" text-anchor="middle" fill="#637581" font-size="11">Nested-CV R²</text>`;s.innerHTML=o}
function deltaChart(){const r=selected('deltaSelect'),s=$('deltachart'),W=620,H=390,L=78,R=24,T=20,B=65,all=[];sets.forEach(d=>{const p=r?.sets[d.id]||{};for(const k of ['delta_vs_raw','delta_vs_random'])if(Number.isFinite(p[k]))all.push(p[k])});let mn=Math.min(-.08,...all),mx=Math.max(.08,...all);mn=Math.floor(mn*10)/10;mx=Math.ceil(mx*10)/10;const x=v=>L+(v-mn)/(mx-mn)*(W-L-R),y=i=>T+i*(H-T-B)/sets.length;let o='';for(let v=Math.ceil(mn*5)/5;v<=mx;v+=.2){o+=svgLine(x(v),T,x(v),H-B,v===0?'#344c55':'#e6ecee',v===0?1.5:1);o+=`<text x="${x(v)}" y="${H-B+17}" text-anchor="middle" fill="#637581" font-size="10">${v.toFixed(1)}</text>`}o+=svgLine(x(.05),T,x(.05),H-B,'#ba5551',1.5,'5 4');sets.forEach((d,i)=>{const p=r?.sets[d.id]||{},yy=y(i);o+=`<text x="${L-8}" y="${yy+4}" text-anchor="end" fill="#42545d" font-size="11">${esc(d.label)}</text>`;for(const [k,sd,off,col,title] of [['delta_vs_raw','sd_vs_raw',-4,'#087e83','FM − raw'],['delta_vs_random','sd_vs_random',4,'#d18a25','FM − random']]){const v=p[k];if(!Number.isFinite(v))continue;const xx=x(v),err=Number.isFinite(p[sd])?Math.abs(x(v+p[sd])-xx):0,cy=yy+off;if(err)o+=svgLine(xx-err,cy,xx+err,cy,col,1.5);o+=`<circle cx="${xx}" cy="${cy}" r="4" fill="${col}"><title>${esc(d.label)} · ${title}: ${fmt(v,4)}${Number.isFinite(p[sd])?` ± ${fmt(p[sd],3)} SD`:''}</title></circle>`}});s.innerHTML=o}
function scoreTable(){const rows=records.slice().sort((a,b)=>b.step-a.step||a.patches-b.patches||a.seed-b.seed);$('scoreTable').innerHTML='<thead><tr><th>Run / checkpoint</th><th>Wins</th><th>Strong</th><th>Mean ΔR²</th><th>FM vs random (mean)</th><th>Evidence</th></tr></thead><tbody>'+rows.map(r=>{const dr=sets.map(d=>r.sets[d.id]?.delta_vs_random).filter(Number.isFinite),mean=dr.length?dr.reduce((a,b)=>a+b,0)/dr.length:null;return `<tr><td>${esc(r.label)}</td><td>${r.wins??'—'}/8</td><td>${r.strong_wins??'—'}</td><td>${fmt(r.mean_delta)}</td><td>${fmt(mean)}</td><td><span class="pill ${r.family==='48-patch'?'no':''}">${esc(r.status)}</span></td></tr>`}).join('')+'</tbody>'}
function screen(){if(!$('screenDecision'))return;const d=D.screen8?.decisions||{},arms=D.screen8?.arms||{};const rows=['random50','block75'].map(k=>{const v=d[k];if(!v)return `<div class="gate"><b>${k}</b><span>Decision data unavailable</span></div>`;const a0=arms[k+'_s0']||{},a1=arms[k+'_s1']||{};const gates=Object.entries(v.gates||{}).map(([name,ok])=>`<div class="gate"><b>${esc(name.replaceAll('_',' '))}</b><span class="${ok?'good':'bad'}">${ok?'PASS':'NOT MET'}</span></div>`).join('');return `<div class="card"><div class="section-head"><div><h3 style="margin:0">${k}</h3><p class="sub">${a0.wins??'—'}/8 and ${a1.wins??'—'}/8 wins · mean gain vs matched control ${fmt(v.mean_delta_vs_control)}</p></div><span class="pill no">NOT PROMOTED</span></div><div class="gate-list">${gates}</div></div>`}).join('');$('screenDecision').innerHTML=`<div class="twocol">${rows}</div><p class="foot">Final assessor: all six arms complete; no missing arms. Promotion gates are experimental triage, not a statistical-significance test.</p>`}
function activeStudy(){if(!$('activeStudySection'))return;const q=D.study;if(!q||!Array.isArray(q.runs)||!q.runs.length){$('activeStudySection').hidden=true;return}$('studySnapshot').textContent=`Status snapshot · ${q.snapshot_at||'time unavailable'} · target ${Number(q.target_steps||497990).toLocaleString()} optimizer steps`;$('activeRuns').innerHTML=q.runs.map(r=>{const id=String(r.run_id||'');const link=/^[a-z0-9]{8}$/.test(id)?`<a href="https://wandb.ai/hcohen/spectralfm-lejepa/runs/${id}" target="_blank" rel="noopener noreferrer">W&amp;B ${esc(id)}</a>`:`W&amp;B ${esc(id||'pending')}`;return `<article class="card metric"><div class="label">${Number(r.patches)===48?'48':'24'} patches · seed ${Number(r.seed)}</div><strong>${Number(r.step||0).toLocaleString()} / ${Number(q.target_steps||497990).toLocaleString()}</strong><small>${esc(r.state||'status unavailable')} · CUDA ${esc(r.cuda_index??'—')} · ${link}</small></article>`}).join('')}
options('recordSelect');options('deltaSelect');$('recordSelect').value=String(Math.max(0,records.findIndex(r=>r.family==='24-patch'&&r.step===497990&&r.seed===0)));$('deltaSelect').value=$('recordSelect').value;$('recordSelect').addEventListener('change',r2chart);$('deltaSelect').addEventListener('change',deltaChart);timeline();r2chart();deltaChart();scoreTable();screen();activeStudy();
})();
</script></body></html>'''.replace('__DATA__', embedded).replace('__STORY__', story_html).replace('__MATRIX__', matrix_html).replace('__VARIANTS__', variants_html).replace('__FLOW__', flow_html).replace('__CANDIDATE__', candidate_html).replace('</style>', '.story-plot{width:100%;height:auto;display:block;margin:14px 0}.story-nav{display:flex;gap:16px;flex-wrap:wrap}.story-nav a{color:#087e83;text-decoration:none;font-weight:650}.pipeline{display:flex;gap:14px;align-items:center}.pipeline div{flex:1;background:#edf7f3;border-radius:10px;padding:16px}.pipeline small{display:block;color:#637581;margin-top:8px}@media(max-width:850px){.pipeline{flex-direction:column;align-items:stretch}.pipeline>span{text-align:center;transform:rotate(90deg)}}html{scroll-behavior:smooth}</style>')
    rendered = re.sub(r'<section class="section" id="activeStudySection">.*?</section>', '', rendered, flags=re.S)
    rendered = re.sub(r'<section class="section"><div class="section-head"><div><h2>Screen 8: masking decision</h2>.*?</section>', '', rendered, flags=re.S)
    stopping = '<section class="section"><details class="card"><summary>Does longer training help? Historical evidence and limits</summary><p>In the legacy 24-patch continuation runs, both seeds won 6/8 datasets at 250,000 steps, compared with 5/8 and 3/8 at the final 497,990-step checkpoint. Downstream performance was not monotonic with training duration.</p><p>This motivates a controlled early-stopping experiment. These are retrospective downstream evaluation scores: choosing a checkpoint from them would reuse the evaluation for selection. A valid stopping rule must be fixed using an independent validation criterion and tested on held-out tasks. The fresh paired study evaluated only final checkpoints, so it cannot establish when those runs should have stopped.</p></details></section>'
    legacy = {(r['seed'], r['step']): r for r in data.get('records', [])
              if r['family'] == '24-patch' and r.get('cohort') != 'fresh paired study'}
    if not all(key in legacy for key in ((0, 250000), (1, 250000), (0, 497990), (1, 497990))):
        stopping = ''
    rendered = re.sub(r'<section class="section"><div class="section-head"><div><h2>Historical checkpoint progression</h2>.*?</section>', stopping, rendered, flags=re.S)
    return rendered



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, default=Path('outputs/training-progress/results_report.html'))
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else args.repo_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    data = collect_report_data(args.repo_root)
    rendered = render_html(data, export_dir=output.parent / 'figures')
    if data.get('matrix', {}).get('complete'):
        rendered = rendered.replace('Confirmatory comparison in progress', 'Final paired comparison completed')
        rendered = rendered.replace('Final eight-dataset evaluations are serialized on the reserved evaluation GPU.', 'All six final checkpoints have completed eight-dataset evaluations.')
        rendered = re.sub(r'<section class="section"><div class="section-head"><div><h2>Current result</h2>.*?id="metrics"></div></section>', '<div id="metrics" hidden></div>', rendered, flags=re.S)
        rendered = rendered.replace('One 48-patch seed and two 24-patch seeds do not establish which architecture is better.', 'The final paired comparison includes three matched training seeds per architecture.')
        rendered = rendered.replace('The existing 48-patch point is one seed at three epochs and is not a long-run comparison.', 'Final paired checkpoints are at 497,990 steps; earlier three-epoch points remain historical context.')
        rendered = re.sub(r'<details class="card" style="margin-top:14px"><summary>Next experiment and decision rule</summary>.*?</details>', '<details class="card" style="margin-top:14px"><summary>Registered final decision</summary><p>Both recipes pass the majority rule: positive seed-mean gain on at least five datasets, and at least two of three seeds individually win five or more. The 48-patch recipe passes the architecture preference rule: positive mean paired gain, winning in two of three seed pairs. This is exploratory recipe selection. The all-eight target remains unmet.</p></details>', rendered, flags=re.S)
    output.write_text(rendered)
    from scripts.results_matrix import matrix_csv
    (output.parent / 'final_results_matrix.csv').write_text(matrix_csv(data['matrix']['final_rows']))
    (output.parent / 'readout_variants.csv').write_text(matrix_csv([*data['variants']['raw_rows'], *data['variants']['rows'], *data['variants']['controls']]))
    archive_rows = [{**r, 'label': f'{e["id"]} · {r["label"]}'} for e in data['matrix']['evaluations'] for r in e['rows']]
    (output.parent / 'all_backbones_results_matrix.csv').write_text(matrix_csv(archive_rows))
    from scripts.architecture_diagram import render_dataflow
    (output.parent / 'tensor_dataflow.svg').write_text(render_dataflow())
    (output.parent / 'results_matrix.json').write_text(json.dumps(data['matrix'], indent=2, allow_nan=False))
    print(f'Report written: {output}')


if __name__ == '__main__':
    main()
