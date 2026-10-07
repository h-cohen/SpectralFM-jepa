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
            data = json.loads(path.read_text())
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


def seed_comparison_figure(rows):
    selected = []
    for seed in (0, 1):
        candidates = [r for r in rows if f'/lr50_s{seed}_' in r['name'] and r['step'] is not None]
        if candidates:
            selected.append(max(candidates, key=lambda r: r['step']))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, row in zip(axes, selected[:2]):
        x = np.arange(8)
        sets = row['sets']
        ax.bar(x-.25, [sets[s]['raw_r2'] for s in SETS], .25, label='Raw spectrum')
        ax.bar(x, [sets[s]['embedding_r2'] for s in SETS], .25, label='Pretrained FM')
        ax.bar(x+.25, [sets[s].get('random_control', {}).get('embedding_r2', np.nan) for s in SETS], .25, label='Random init')
        ax.set_xticks(x, [s.replace('dataset','') for s in SETS], rotation=45, ha='right')
        ax.set_title(f"{row['name'].split('/')[-1]}: {row['wins']}/8 wins")
        ax.set_ylabel('Nested-CV R²'); ax.legend(fontsize=8)
    fig.suptitle('Latest evaluated 50% mix checkpoints (not current live weights)')
    fig.tight_layout()
    return fig


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
<p><b>Scoring:</b> frozen nested-CV parameter_0 regression; 2×5 outer, 5 inner, seed 42.
Any ΔR²&gt;0 wins; ΔR²≥.05 is strong. Normalizers/readouts/probes are selected inside folds.
Pretraining includes evaluation spectra without labels (approved transductive setup).
Tiny-set estimates vary substantially; present repeated seed patterns and paired uncertainty.
A result below is a measured checkpoint result, never a live-weight estimate.</p>
<table border="1" cellpadding="6"><tr><th>Experiment/checkpoint</th><th>Step</th><th>Wins</th><th>Strong</th><th>Mean raw R²</th><th>Mean model R²</th><th>Mean ΔR²</th><th>Mean random R²</th></tr>{table}</table>
<p>Missing random controls are left blank in the scorecard table. Seed variability is not a confidence interval.
All-eight has not been reached. Evaluation checkpoint results update when completed scorecards appear.</p>
</body></html>'''


def main():
    import wandb
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--watch', action='store_true')
    ap.add_argument('--interval', type=float, default=60)
    args = ap.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=60)
    sources = {}
    for seed, stamp, resumed in [(0, '100813', 'wxrdfq4n'), (1, '100812', 'wuw24e91')]:
        cfg = json.loads((ROOT / f'outputs/long-1/eval/lr50_s{seed}_20261007-{stamp}_step200000/summary.json').read_text())
        sources[seed] = [cfg['lineage']['pretraining_run_id'], resumed]
    cfg = __import__('yaml').safe_load((ROOT / 'configs/long1_resume_s0.yaml').read_text())
    total = 497990
    run = wandb.init(entity='hcohen', project='spectralfm-lejepa', group='training-presentation',
                     job_type='presentation', name='Training procedure — live two-seed progress',
                     tags=['presentation', 'live-progress', 'no-artifacts'],
                     config={'source_runs': sources, 'training': cfg['training'], 'target_steps': total,
                             'goal': 'frozen FM beats raw on majority/all eight', 'protocol': 'unchanged nested CV'})
    (OUTPUT / 'run.json').write_text(json.dumps({'id': run.id, 'url': run.url, 'sources': sources}, indent=2))
    print('Presentation URL:', run.url, flush=True)
    for seed in sources:
        run.define_metric(f'seed{seed}/optimizer_step')
        run.define_metric(f'seed{seed}/*', step_metric=f'seed{seed}/optimizer_step')
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
                    last[seed] = {'step':step, 'source_run':rid, 'state':source.state}
                run.summary[f'seed{seed}/source_url'] = source.url
        roots = [ROOT/'outputs', ROOT.parent/'SpectralFM-jepa-wt'/'outputs']
        rows = evaluation_rows([root for root in roots if root.exists()])
        new_signature = [(r['path'], Path(r['path']).stat().st_mtime_ns) for r in rows]
        if rows and new_signature != signature:
            signature = new_signature
            columns, values = evaluation_table(rows)
            report = narrative(rows)
            (OUTPUT/'presentation.html').write_text(report)
            (OUTPUT/'scorecards.json').write_text(json.dumps(rows, indent=2))
            run.log({'procedure/guide':wandb.Html(report),
                     'evaluation/all_datasets':wandb.Table(columns=columns, data=values)})
            for key, fig in [('evaluation/experiment_comparison', comparison_figure(rows)),
                             ('evaluation/latest_two_seeds', seed_comparison_figure(rows))]:
                fig.savefig(OUTPUT/(key.split('/')[-1]+'.png'), dpi=180, bbox_inches='tight')
                run.log({key:wandb.Image(fig)}); plt.close(fig)
            run.summary['evaluation/complete_scorecards'] = len(rows)
            run.summary['evaluation/best_observed_wins'] = max(r['wins'] for r in rows)
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
