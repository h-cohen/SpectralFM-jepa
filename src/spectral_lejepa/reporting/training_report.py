"""Figures for docs/TRAINING_REPORT.md from saved outputs only (no GPU).

Inputs: outputs/training-report/wandb_history.json (W&B history export), the robust-ensemble and
robustness-continuation outputs. Writes SVGs to docs/figures/training-report/.
    bash -ic 'python -m spectral_lejepa.reporting.training_report fetch'   # W&B key from the user shell
    python -m spectral_lejepa.reporting.training_report
    python -m spectral_lejepa.reporting.training_report_html   # docs/TRAINING_REPORT.html
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from spectral_lejepa.evaluation.ensemble import ensemble_predictions, member_oof as job_oof
from spectral_lejepa.evaluation.perturbation import DATASETS
from spectral_lejepa.evaluation.nested import r2

ROOT = Path('outputs/robust-ensemble')
OUT = Path('docs/figures/training-report')
HIST = Path('outputs/training-report/wandb_history.json')
PILOT = Path('outputs/robustness-continuation')
SHORT = {d: d.replace('dataset', '') for d in DATASETS}
# Block2-flat scores of the shortlisted models (docs/PRODUCTION_RECOMMENDATION.md, results_analysis.json files).
CANDIDATES = [('seed101\n(export)', 5, 6, .0801), ('48p seed2', 7, 6, .0894), ('ensemble\n0/1/2/101', 7, 6, .0858),
              ('control\ns401', 7, 6, .0846), ('control\ns402', 7, 6, .0871), ('robust\ns401', 7, 8, .0436),
              ('robust\ns402', 8, 8, .0469), ('robust\nensemble', 8, 8, .0556)]
RUNS = {'s101_full': 'ylymmcnn', 'p48_s0': 'kknz92e3', 'p48_s1': 'm1p2atdu', 'p48_s2': 'ceanfxq1',
        'control_s401': 'c3g6saaz', 'robust_s401': 'bqw877zh',
        'robust_p0': '8nfxzsde', 'robust_p1': 'ya6s9o5p', 'robust_s101': 'ok3up62j'}
KEYS = ('train/loss', 'train/mse_loss', 'train/sigreg_loss', 'train/global_loss', 'train/robust_loss',
        'train/learning_rate', 'valid/mse_loss', 'representation/token_effective_rank')
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
                     'svg.hashsalt': 'spectralfm'})   # reproducible SVG ids


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, metadata={'Date': None})
    plt.close(fig)


def curve(ax, h, key, label, smooth=25, **kw):
    if key not in h:
        return
    s, v = np.array(h[key], dtype=float).T
    if smooth > 1 and len(v) > smooth:
        v = np.convolve(v, np.ones(smooth) / smooth, mode='valid')
        s = s[smooth // 2: smooth // 2 + len(v)]
    ax.plot(s, v, label=label, lw=1.2, **kw)


def pretraining_curves(hist):
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6))
    panels = [('train/mse_loss', 'masked-prediction MSE', 'log'), ('train/sigreg_loss', 'token SIGReg', 'log'),
              ('train/global_loss', 'pooled-view loss', 'log'), ('train/learning_rate', 'learning rate', 'linear'),
              ('valid/mse_loss', 'validation MSE (10k held-out rows)', 'log'),
              ('representation/token_effective_rank', 'token effective rank (of 256)', 'linear')]
    runs = [('p48_s0', '48p seed0'), ('p48_s1', '48p seed1'), ('p48_s2', '48p seed2'), ('s101_full', 'seed101 final')]
    for ax, (key, title, scale) in zip(axes.flat, panels):
        for name, label in runs:
            curve(ax, hist[name], key, label, smooth=1 if key.startswith(('valid', 'repr', 'train/learning')) else 25)
        ax.set(title=title, xlabel='step', yscale=scale)
    axes[0, 0].legend(frameon=False)
    fig.suptitle('Stage 1: from-scratch pretraining, 497,990 steps (10 epochs)', fontsize=10)
    save(fig, 'fig1_pretraining_curves.svg')


def continuation_curves(hist):
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    panels = [('train/mse_loss', 'masked-prediction MSE'), ('train/robust_loss', 'robust term (unweighted)'),
              ('train/sigreg_loss', 'token SIGReg'), ('representation/token_effective_rank', 'token effective rank')]
    runs = [('control_s401', 'control s401 (weight 0)', '--'), ('robust_s401', 'robust s401', '-'),
            ('robust_p0', 'robust p0', '-'), ('robust_p1', 'robust p1', '-'), ('robust_s101', 'robust s101', '-')]
    for ax, (key, title) in zip(axes, panels):
        for name, label, ls in runs:
            curve(ax, hist[name], key, label, smooth=1 if key.startswith('repr') else 25, ls=ls)
        ax.set(title=title, xlabel='continuation step', yscale='log' if key in ('train/mse_loss', 'train/robust_loss') else 'linear')
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle('Stage 2: robust continuation, 49,799 steps from each saved checkpoint', fontsize=10)
    save(fig, 'fig2_continuation_curves.svg')


def candidates():
    names, raw_w, rnd_w, d = zip(*CANDIDATES)
    x = np.arange(len(names))
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 3.6))
    a.bar(x - .2, raw_w, .4, label='wins vs raw', color='#3a6ea5')
    a.bar(x + .2, rnd_w, .4, label='wins vs random weights', color='#9db4d0')
    a.axhline(5, color='k', lw=.8, ls=':')
    a.text(len(x) - .5, 5.1, 'contract: 5/8', ha='right', fontsize=8)
    a.set(xticks=x, xticklabels=names, ylim=(0, 8.6), ylabel='datasets (of 8)', title='Win counts')
    a.legend(frameon=False, loc='lower left', fontsize=8)
    b.bar(x, d, color=['#888'] * 5 + ['#c0504d'] * 3)
    b.set(xticks=x, xticklabels=names, ylabel='mean ΔR² vs raw', title='Mean margin over raw')
    save(fig, 'fig4_candidates.svg')


def ensemble_scores():
    m = json.loads((ROOT / 'manifest.json').read_text())
    members = [Path(m['seed2_member'])] + [ROOT / n for n in m['jobs']]
    ref = Path('outputs/frozen-feature-study/prior_s2')
    score = lambda y, P: np.mean([r2(y, p) for p in P])
    rows = {}
    for ds in DATASETS:
        o = [job_oof(b, ds) for b in members]
        y = o[0]['y']
        rows[ds] = (score(y, o[0]['raw']), score(y, np.load(ref / ds / 'linear_oof.npz')['random_layer2/flat']),
                    score(y, ensemble_predictions(o)))
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(len(DATASETS))
    for i, (lab, c) in enumerate([('raw spectra', '#999'), ('random-weight encoder', '#9db4d0'),
                                  ('robust ensemble', '#c0504d')]):
        ax.bar(x + (i - 1) * .27, [rows[d][i] for d in DATASETS], .27, label=lab, color=c)
    for xi, d in zip(x, DATASETS):
        ax.text(xi + .27, rows[d][2] + .015, f'{rows[d][2] - rows[d][0]:+.3f}', ha='center', fontsize=7)
    ax.set(xticks=x, xticklabels=[SHORT[d] for d in DATASETS], ylim=(0, 1.08),
           ylabel='nested-CV R² (parameter_0)', title='Robust ensemble vs controls, block2-flat (label: Δ vs raw)')
    ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(.5, -.1), fontsize=8)
    save(fig, 'fig5_ensemble_r2.svg')


def member_heatmap(res):
    names = [Path(k).name for k in res['per_job_delta_raw']] + ['ENSEMBLE']
    M = np.array([[v[d] for d in DATASETS] for v in res['per_job_delta_raw'].values()]
                 + [[res['sets'][d]['vs_raw']['delta'] for d in DATASETS]])
    fig, ax = plt.subplots(figsize=(8, 2.8))
    lim = .16
    ax.imshow(np.clip(M, -lim, lim), cmap='RdBu', vmin=-lim, vmax=lim, aspect='auto')
    for (i, j), v in np.ndenumerate(M):
        ax.text(j, i, f'{v:+.3f}', ha='center', va='center', fontsize=7)
    ax.set(xticks=range(len(DATASETS)), xticklabels=[SHORT[d] for d in DATASETS], yticks=range(len(names)),
           yticklabels=names, title='ΔR² vs raw per member: averaging removes every member loss')
    ax.axhline(len(names) - 1.5, color='k', lw=1)
    save(fig, 'fig6_member_deltas.svg')


def robustness(res):
    pert = ['noise_1pct', 'noise_3pct', 'offset_5pct', 'slope_5pct', 'scale_1.05', 'shift_+1', 'shift_2',
            'noise_5pct', 'sine_baseline_5pct']
    pairs = [(PILOT / 'start_perturbation.json', PILOT / 'robust_s401' / 'perturbation.json')]
    pairs += [(ROOT / n / 'start_perturbation.json', ROOT / n / 'perturbation.json') for n in res['drift']]
    load = lambda p: json.loads(p.read_text())['summary']
    S, J = [load(s) for s, _ in pairs], [load(j) for _, j in pairs]
    med = lambda L, part, k: np.median([s[part][k] for s in L])
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.4))
    x = np.arange(len(pert))
    a.bar(x - .2, [med(S, 'drift_median_of_medians', k) for k in pert], .4, label='start checkpoints', color='#999')
    a.bar(x + .2, [med(J, 'drift_median_of_medians', k) for k in pert], .4, label='robust continuations', color='#c0504d')
    a.set(xticks=x, xticklabels=pert, yscale='log', ylabel='relative block2 feature drift',
          title='Feature drift (median of 4 members)')
    plt.setp(a.get_xticklabels(), rotation=40, ha='right')
    a.legend(frameon=False, fontsize=8)
    keys = ['clean'] + pert
    x = np.arange(len(keys))
    b.bar(x - .2, [med(S, 'mean_r2_small_sets', k) for k in keys], .4, color='#999')
    b.bar(x + .2, [med(J, 'mean_r2_small_sets', k) for k in keys], .4, color='#c0504d')
    b.set(xticks=x, xticklabels=keys, ylim=(-1, 1), ylabel='mean R² (7 small sets)',
          title='Fixed-ridge R² on perturbed inputs (clipped at −1)')
    plt.setp(b.get_xticklabels(), rotation=40, ha='right')
    b.axvline(6.5, color='k', lw=.6, ls=':')
    b.text(6.6, .75, 'held out\nfrom training', fontsize=7)
    save(fig, 'fig7_robustness.svg')


def fetch():
    import wandb
    api, out = wandb.Api(), {}
    for name, rid in RUNS.items():
        run = api.run(f'hcohen/spectralfm-lejepa/{rid}')
        out[name] = {'id': rid}
        for k in KEYS:
            rows = run.history(samples=1500, keys=['train/global_step', k], pandas=False)
            if rows:
                out[name][k] = [[r['train/global_step'], r[k]] for r in rows]
    HIST.parent.mkdir(parents=True, exist_ok=True)
    HIST.write_text(json.dumps(out))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hist = json.loads(HIST.read_text())
    res = json.loads((ROOT / 'results_analysis.json').read_text())
    pretraining_curves(hist)
    continuation_curves(hist)
    candidates()
    ensemble_scores()
    member_heatmap(res)
    robustness(res)


if __name__ == '__main__':
    fetch() if sys.argv[1:] == ['fetch'] else main()
