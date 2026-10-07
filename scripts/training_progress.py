"""Publish a live W&B presentation from source histories and measured probe scorecards.

The source training runs are read-only. This creates a separate presentation run.
Usage: bash -ic '.venv/bin/python -m scripts.training_progress --watch'
"""
from __future__ import annotations

import argparse
import html
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.scorecard_figures import dataset_comparison_figure, paired_gains_figure
from spectral_lejepa.utils.wandb import training_metrics

SETS = ('labeled_data', 'dataset0055', 'dataset0106', 'dataset0109',
        'dataset0112', 'dataset0113', 'dataset0114', 'dataset0120')
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs' / 'training-progress'


def evaluation_rows(roots):
    """Only full eight-set scorecards count toward the stated goal."""
    rows = []
    for root in roots:
        for path in sorted(Path(root).rglob('summary.json')):
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            if not set(SETS).issubset(data.get('sets', {})):
                continue
            sets = {name: data['sets'][name] for name in SETS}
            if not all('embedding_r2' in s and 'raw_r2' in s for s in sets.values()):
                continue
            deltas = [s['embedding_r2'] - s['raw_r2'] for s in sets.values()]
            rows.append(dict(name=f'{path.parent.parent.parent.name}/{path.parent.name}',
                             path=str(path), step=data.get('lineage', {}).get('pretraining_checkpoint_step'),
                             sets=sets, wins=sum(d > 0 for d in deltas),
                             strong_wins=sum(d >= .05 for d in deltas), mean_delta=float(np.mean(deltas))))
    return rows


def load_sources(path):
    """Load source run IDs from a manifest keyed by seed number."""
    manifest = json.loads(Path(path).read_text())
    if not isinstance(manifest, dict):
        raise ValueError('source manifest must be a JSON object')
    sources = {}
    for seed in (0, 1):
        key = str(seed)
        if key not in manifest:
            raise ValueError(f'source manifest is missing seed {seed}')
        run_ids = manifest[key]
        if not isinstance(run_ids, list) or not run_ids:
            raise ValueError(f'source manifest seed {seed} must have a non-empty list of run IDs')
        if any(not isinstance(run_id, str) or not run_id for run_id in run_ids):
            raise ValueError(f'source manifest seed {seed} run IDs must be non-empty strings')
        sources[seed] = run_ids
    return sources


def history_cursors(api, sources, last_status, processed_cursors=None):
    """Resume source scans after rows already represented on the live run."""
    cursors = {}
    processed_cursors = processed_cursors or {}
    for seed, run_ids in sources.items():
        latest = last_status.get(str(seed), {})
        latest_run = latest.get('source_run')
        latest_step = latest.get('step')
        for run_id in run_ids:
            if run_id in processed_cursors:
                cursors[run_id] = int(processed_cursors[run_id])
                continue
            if run_id == latest_run and latest_step is not None:
                history_step = latest.get('history_step')
                if history_step is None:
                    source = api.run(f'hcohen/spectralfm-lejepa/{run_id}')
                    history_step = -1
                    for row in source.scan_history(keys=['train/global_step'], page_size=50000):
                        if row.get('train/global_step') == latest_step and row.get('_step') is not None:
                            history_step = max(history_step, int(row['_step']))
                cursors[run_id] = int(history_step)
                continue
            source = api.run(f'hcohen/spectralfm-lejepa/{run_id}')
            cursor = source.summary.get('_step', source.summary.get('train/global_step', -1))
            cursors[run_id] = int(cursor) if cursor is not None else -1
    return cursors


def evaluation_table(rows):
    columns = ['experiment', 'checkpoint_step', 'dataset', 'n', 'raw_r2', 'model_r2',
               'delta_vs_raw', 'paired_delta_sd', 'random_r2', 'delta_vs_random',
               'paired_delta_vs_random_sd', 'prob_model_beats_random', 'wins_of_8']
    values = []
    for row in rows:
        for name, s in row['sets'].items():
            random = s.get('random_control', {}).get('embedding_r2')
            values.append([row['name'], row['step'], name, s.get('n'), s['raw_r2'],
                           s['embedding_r2'], s['embedding_r2'] - s['raw_r2'],
                           s.get('embedding_minus_raw', {}).get('sd'), random,
                           s['embedding_r2'] - random if random is not None else None,
                           s.get('vs_random_control', {}).get('sd'),
                           s.get('vs_random_control', {}).get('p_a_better'), row['wins']])
    return columns, values


def comparison_figure(rows):
    # Show all measured checkpoints; recipe differences are explicitly named, not a time curve.
    fig, ax = plt.subplots(figsize=(12, max(3, .38 * len(rows) + 1.5)))
    values = np.array([[r['sets'][s]['embedding_r2'] - r['sets'][s]['raw_r2'] for s in SETS] for r in rows])
    bound = max(.1, min(.5, float(np.max(np.abs(values)))))
    image = ax.imshow(values, cmap='RdBu', vmin=-bound, vmax=bound, aspect='auto')
    ax.set_xticks(range(8), [s.replace('dataset', '') for s in SETS])
    ax.set_yticks(range(len(rows)), [f"{r['name']} ({r['wins']}/8)" for r in rows], fontsize=8)
    for i in range(len(rows)):
        for j in range(8):
            ax.text(j, i, f'{values[i,j]:+.3f}', ha='center', va='center', fontsize=7)
    ax.set_title('Measured frozen-probe ΔR² vs raw — different recipes/checkpoints')
    fig.colorbar(image, ax=ax, label='Model R² − raw R²')
    fig.tight_layout()
    return fig


def latest_seed_rows(rows):
    selected = []
    for seed in (0, 1):
        candidates = [r for r in rows if f'/lr50_s{seed}_' in r['name'] and r['step'] is not None]
        if candidates:
            selected.append(max(candidates, key=lambda r: r['step']))
    return selected


def _latest_figure_records(rows):
    records = []
    for row in latest_seed_rows(rows):
        sets = {}
        for name, score in row['sets'].items():
            sets[name] = {
                'n': score.get('n'), 'raw_r2': score.get('raw_r2'),
                'model_r2': score.get('embedding_r2'),
                'random_r2': score.get('random_control', {}).get('embedding_r2'),
                'delta_vs_raw': score.get('embedding_r2') - score.get('raw_r2'),
                'sd_vs_raw': score.get('embedding_minus_raw', {}).get('sd'),
                'delta_vs_random': (score.get('embedding_r2') - score['random_control']['embedding_r2']
                                    if score.get('random_control', {}).get('embedding_r2') is not None else None),
                'sd_vs_random': score.get('vs_random_control', {}).get('sd'),
            }
        seed_label = 0 if '/lr50_s0_' in row['name'] else 1
        records.append({'label': f'seed{seed_label} step {row["step"]}', 'sets': sets})
    return records


def seed_comparison_figure(rows):
    records = _latest_figure_records(rows)
    steps = ', '.join(record['label'] for record in records) or 'no completed checkpoints'
    fig = dataset_comparison_figure(
        records, SETS, title=f'Latest evaluated 50% mix checkpoints ({steps}; not current live weights)')
    return fig


def seed_paired_gains_figure(rows):
    records = _latest_figure_records(rows)
    steps = ', '.join(record['label'] for record in records) or 'no completed checkpoints'
    return paired_gains_figure(
        records, SETS, title=f'Latest evaluated 50% mix paired gains ({steps}; not current live weights)')


def narrative(rows):
    def mean_score(row, key):
        values = [s.get('random_control', {}).get('embedding_r2') if key == 'random'
                  else s[key] for s in row['sets'].values()]
        return f'{np.mean(values):.4f}' if all(v is not None for v in values) else 'not measured'
    table = ''.join(f"<tr><td>{html.escape(r['name'])}</td><td>{r['step']}</td><td>{r['wins']}/8</td>"
                    f"<td>{r['strong_wins']}</td><td>{mean_score(r, 'raw_r2')}</td>"
                    f"<td>{mean_score(r, 'embedding_r2')}</td><td>{r['mean_delta']:+.4f}</td>"
                    f"<td>{mean_score(r, 'random')}</td></tr>" for r in rows)
    return f'''<html><body style="font-family:Arial;max-width:1100px;margin:24px">
<h1>SpectralFM: training procedure and measured progress</h1>
<p><b>Goal:</b> one frozen embedding beating raw spectra on a majority, ideally all eight datasets.</p>
<p>Current experiment: 24-patch shared-encoder LeJEPA, global affine normalization, mask 75%,
SIGReg λ=.05, pooled-view objective weight 1, 50% labeled-regression-source mix, two seeds, 10 epochs.
Training restarted from step 200,000 with restored AdamW, scaler and original cosine schedule.
Legacy checkpoints lacked RNG state, so exact uninterrupted stochastic replay is unavailable.</p>
<p><b>Reading this view:</b> seed0/seed1 curves join the original and resumed runs on absolute optimizer step.
Total loss = masked prediction MSE + .05 × token SIGReg + pooled-view loss.
The pooled loss is .95 × view invariance + .05 × pooled SIGReg. Weighted contributions are logged separately.
Validation uses fixed spectra/masks every 25,000 steps; representation rank/std diagnose collapse.
Lower pretraining loss does not establish downstream improvement.</p>
<p><b>Screen 8:</b> six matched continuations compare random75, random50 and block75 across seeds0/1. Each initializes from its seed’s 200k checkpoint, resets AdamW equally, and runs a fresh 15k-step schedule. Screen-step15000 means 200k initial exposure plus15k continuation; it is not a from-scratch comparison. Validation/diagnostics occur every5k in this screen. Candidate selection requires repeated hard-set improvement with majority and regression guards; final recipe requires longer confirmation and a third seed.</p>
<p><b>Scoring:</b> frozen nested-CV parameter_0 regression; 2×5 outer, 5 inner, seed 42.
Any ΔR²&gt;0 wins; ΔR²≥.05 is strong. Normalizers/readouts/probes are selected inside folds.
Pretraining includes evaluation spectra without labels (approved transductive setup).
Tiny-set estimates vary substantially; present repeated seed patterns and paired uncertainty.
A result below is a measured checkpoint result, never a live-weight estimate.</p>
<table border="1" cellpadding="6"><tr><th>Experiment/checkpoint</th><th>Step</th><th>Wins</th><th>Strong</th><th>Mean raw R²</th><th>Mean model R²</th><th>Mean ΔR²</th><th>Mean random R²</th></tr>{table}</table>
<p>Missing random controls are left blank in the scorecard table. Seed variability is not a confidence interval.
All-eight has not been reached. Evaluation checkpoint results update when completed scorecards appear.</p>
</body></html>'''


def publish_evaluations(run, rows):
    import wandb
    columns, values = evaluation_table(rows)
    report = narrative(rows)
    (OUTPUT / 'presentation.html').write_text(report)
    (OUTPUT / 'scorecards.json').write_text(json.dumps(rows, indent=2))
    run.log({'procedure/guide': wandb.Html(report),
             'evaluation/all_datasets': wandb.Table(columns=columns, data=values)})
    for key, fig in [('evaluation/experiment_comparison', comparison_figure(rows)),
                     ('evaluation/latest_per_dataset_r2', seed_comparison_figure(rows)),
                     ('evaluation/latest_paired_gains', seed_paired_gains_figure(rows))]:
        fig.savefig(OUTPUT / (key.split('/')[-1] + '.png'), dpi=180, bbox_inches='tight')
        run.log({key: wandb.Image(fig)})
        plt.close(fig)
    run.summary['evaluation/complete_scorecards'] = len(rows)
    run.summary['evaluation/best_observed_wins'] = max(r['wins'] for r in rows)


def init_presentation_run(wandb, sources, cfg, total_steps):
    options = {
        'entity': 'hcohen', 'project': 'spectralfm-lejepa', 'group': 'training-presentation',
        'job_type': 'presentation', 'name': 'Training procedure — live two-seed progress',
        'tags': ['presentation', 'live-progress', 'no-artifacts'],
        'config': {'source_runs': sources, 'training': cfg['training'], 'target_steps': total_steps,
                   'goal': 'frozen FM beats raw on majority/all eight', 'protocol': 'unchanged nested CV'},
    }
    manifest_path = OUTPUT / 'run.json'
    resuming = False
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get('id'):
            options.update(id=previous['id'], resume='allow')
            resuming = True
    run = wandb.init(**options)
    manifest_path.write_text(json.dumps({'id': run.id, 'url': run.url, 'sources': sources}, indent=2))
    return run, resuming


def main(argv=None):
    import wandb
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--watch', action='store_true')
    ap.add_argument('--interval', type=float, default=60)
    ap.add_argument('--source-manifest', type=Path)
    ap.add_argument('--refresh-evaluations', action='store_true',
                    help='update evaluation tables and figures on the existing presentation run')
    args = ap.parse_args(argv)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.refresh_evaluations:
        run_info = json.loads((OUTPUT / 'run.json').read_text())
        run = wandb.init(entity='hcohen', project='spectralfm-lejepa', id=run_info['id'], resume='allow')
        roots = [ROOT / 'outputs', ROOT.parent / 'SpectralFM-jepa-wt' / 'outputs']
        rows = evaluation_rows([root for root in roots if root.exists()])
        if rows:
            publish_evaluations(run, rows)
        run.finish()
        print(f'Refreshed evaluation figures for {len(rows)} complete scorecards on {run.url}', flush=True)
        return
    api = wandb.Api(timeout=60)
    if args.source_manifest:
        sources = load_sources(args.source_manifest)
    else:
        sources = {}
        for seed, stamp, resumed in [(0, '100813', 'wxrdfq4n'), (1, '100812', 'wuw24e91')]:
            cfg = json.loads((ROOT / f'outputs/long-1/eval/lr50_s{seed}_20261007-{stamp}_step200000/summary.json').read_text())
            sources[seed] = [cfg['lineage']['pretraining_run_id'], resumed]
    cfg = __import__('yaml').safe_load((ROOT / 'configs/long1_resume_s0.yaml').read_text())
    total = 497990
    run, resuming = init_presentation_run(wandb, sources, cfg, total)
    print('Presentation URL:', run.url, flush=True)
    for seed in sources:
        run.define_metric(f'seed{seed}/optimizer_step')
        run.define_metric(f'seed{seed}/*', step_metric=f'seed{seed}/optimizer_step')
    if resuming:
        status_path = OUTPUT / 'status.json'
        last_status = json.loads(status_path.read_text()) if status_path.exists() else {}
        cursor_path = OUTPUT / 'cursors.json'
        processed_cursors = json.loads(cursor_path.read_text()) if cursor_path.exists() else {}
        cursors = history_cursors(api, sources, last_status, processed_cursors)
    else:
        cursors = {rid: -1 for ids in sources.values() for rid in ids}
    signature = None
    last = {}
    while True:
        for seed, ids in sources.items():
            for rid in ids:
                source = api.run(f'hcohen/spectralfm-lejepa/{rid}')
                # Historical terminal checkpoint coordinates are authoritative.
                for row in source.scan_history(page_size=50000, min_step=cursors[rid]+1):
                    step = row.get('train/global_step', row.get('_step'))
                    cursors[rid] = max(cursors[rid], row['_step'])
                    if step is None or step > total:
                        continue
                    measured = {k:v for k,v in row.items() if k.startswith(('train/', 'valid/', 'representation/'))
                                and isinstance(v,(int,float)) and math.isfinite(v)}
                    if not measured:
                        continue
                    measured = training_metrics(measured, cfg['training'], total)
                    payload = {f'seed{seed}/{k}':v for k,v in measured.items()}
                    payload[f'seed{seed}/optimizer_step'] = step
                    run.log(payload)
                    last[seed] = {'step':step, 'history_step':row['_step'],
                                  'source_run':rid, 'state':source.state}
                run.summary[f'seed{seed}/source_url'] = source.url
                (OUTPUT / 'cursors.json').write_text(json.dumps(cursors, indent=2))
        roots = [ROOT/'outputs', ROOT.parent/'SpectralFM-jepa-wt'/'outputs']
        rows = evaluation_rows([root for root in roots if root.exists()])
        new_signature = [(r['path'], Path(r['path']).stat().st_mtime_ns) for r in rows]
        if rows and new_signature != signature:
            signature = new_signature
            publish_evaluations(run, rows)
        run.summary['last_refresh_unix'] = time.time()
        (OUTPUT/'status.json').write_text(json.dumps(last, indent=2))
        print('Refresh:', json.dumps(last), 'scorecards:', len(rows), flush=True)
        if not args.watch:
            break
        time.sleep(args.interval)
        api.flush()
    run.finish()


if __name__ == '__main__':
    main()
