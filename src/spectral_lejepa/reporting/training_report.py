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
from spectral_lejepa.reporting.architecture import block_diagram

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


def ensemble_oof():
    """{set: {'y', 'raw spectra': (repeats, N), 'robust ensemble': (repeats, N)}}, block2-flat nested OOF."""
    m = json.loads((ROOT / 'manifest.json').read_text())
    members = [Path(m['seed2_member'])] + [ROOT / n for n in m['jobs']]
    out = {}
    for ds in DATASETS:
        o = [job_oof(b, ds) for b in members]
        out[ds] = {'y': o[0]['y'], 'raw spectra': o[0]['raw'], 'robust ensemble': ensemble_predictions(o)}
    return out


def true_vs_predicted(preds, name='fig11_true_vs_predicted.svg', panel=1.5):
    """One row per set, one column per model, in label units. Each row's axes span the set's label range;
    predictions outside it are clipped to the edge. Points are the mean over outer-CV repeats; R² is the
    reported value (mean over repeats)."""
    sets = list(preds)
    cols = [c for c in preds[sets[0]] if c != 'y']
    colors = dict(zip(cols, ('#3a6ea5', '#c0504d', '#2a9d8f', '#e9a23b')))
    fig, axes = plt.subplots(len(sets), len(cols), figsize=(panel * len(cols) + .7, panel * len(sets) + .3),
                             squeeze=False)
    best = {}
    for i, ds in enumerate(sets):
        y = preds[ds]['y']
        pad = .08 * (y.max() - y.min())
        lo, hi = y.min() - pad, y.max() + pad
        scores = {c: float(np.mean([r2(y, p) for p in preds[ds][c]])) for c in cols}
        best[ds] = max(scores, key=scores.get)
        for j, c in enumerate(cols):
            ax = axes[i, j]
            ax.scatter(y, np.clip(preds[ds][c].mean(0), lo, hi), s=3, alpha=.5, color=colors[c], lw=0)
            ax.plot([lo, hi], [lo, hi], 'k--', lw=.6)
            ax.set(xlim=(lo, hi), ylim=(lo, hi), aspect='equal')
            ax.tick_params(labelsize=6, length=2)
            ax.text(.04, .96, f'R² = {scores[c]:.3f}', transform=ax.transAxes, fontsize=6.5, va='top', weight='bold')
            if c == best[ds]:
                for s in ax.spines.values():
                    s.set(visible=True, color='#3a6ea5', lw=1.5)
            if i == 0:
                ax.set_title(c, fontsize=8, weight='bold')
            if j == 0:
                ax.set_ylabel(f'{SHORT.get(ds, ds)}\npredicted', fontsize=7)
            if i == len(sets) - 1:
                ax.set_xlabel('true', fontsize=7)
    save(fig, name)
    return best


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


MEMBER_EVALS = {'robust_s401': 'outputs/robustness-continuation/robust_s401/eval/*_step49799',
                'robust_p0': 'outputs/robust-ensemble/robust_p0/eval/*_step49799',
                'robust_p1': 'outputs/robust-ensemble/robust_p1/eval/*_step49799',
                'robust_s101': 'outputs/robust-ensemble/robust_s101/eval/*_step49799'}


def _isotropy(X):
    """Effective rank (exp entropy of the centred PCA spectrum) and mean pairwise cosine of rows of X."""
    X = X.astype(np.float64)
    s = np.linalg.svd(X - X.mean(0), compute_uv=False) ** 2
    p = s / s.sum()
    u = X / np.linalg.norm(X, axis=1, keepdims=True)
    return float(np.exp(-(p * np.log(p + 1e-12)).sum())), float((u @ u.T).mean())


def depth_profile(member_evals=None):
    """Block 0-6 flat-readout gain over raw and labeled_data feature isotropy, trained vs random weights."""
    import glob
    member_evals = member_evals or MEMBER_EVALS
    stats = {'trained': {}, 'random': {}}
    for name, pattern in member_evals.items():
        run = Path(sorted(glob.glob(pattern))[0])
        for kind, sub in (('trained', ''), ('random', 'random_control')):
            delta = []
            for k in range(7):
                gains = []
                for ds in DATASETS:
                    raw = json.loads((run / ds / 'nested_results.json').read_text())['families']['raw']['r2_mean']
                    fam = json.loads((run / ds / sub / 'nested_results.json').read_text())['families']
                    gains.append(fam[f'layer{k}/flat']['r2_mean'] - raw)
                delta.append(float(np.mean(gains)))
            bank = np.load(run / 'labeled_data' / sub / 'bank.npz', allow_pickle=True)
            iso = [_isotropy(bank[f'bank__layer{k}'][:, 0, 0]) for k in range(7)]   # mean-pooled tokens
            stats[kind][name] = {'delta': delta, 'effective_rank': [i[0] for i in iso], 'mean_cosine': [i[1] for i in iso]}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    blocks = np.arange(7)
    for ax, key, title in zip(axes, ('delta', 'effective_rank', 'mean_cosine'),
                              ('flat readout: mean ΔR² vs raw (8 sets)', 'labeled_data: effective rank (pooled)',
                               'labeled_data: mean pairwise cosine (pooled)')):
        for kind, color, ls in (('trained', '#c0504d', '-'), ('random', '#888', '--')):
            vals = np.array([v[key] for v in stats[kind].values()])
            for row in vals:
                ax.plot(blocks, row, color=color, lw=.6, alpha=.4, ls=ls)
            ax.plot(blocks, vals.mean(0), color=color, lw=2, ls=ls, marker='o', ms=3,
                    label=f'{kind} weights (mean of members)')
        ax.set(title=title, xlabel='encoder block', xticks=blocks)
    axes[0].axhline(0, color='k', lw=.6)
    axes[0].axvline(2, color='k', lw=.6, ls=':')
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, 'fig8_depth_isotropy.svg')
    return stats


ABLATION = Path('outputs/sigreg-depth-ablation')
PACKED = Path('/mnt5/home/hadar/nova/data/spectra_all')
EXAMPLES = Path('outputs/training-report/input_examples.npz')
EXAMPLE_SOURCES = ('single_channel_all', 'multi_channel', 'labeled_regression/dataset0055')


def input_examples(packed=PACKED, cache=EXAMPLES, seed=0):
    """One seeded spectrum per source in EXAMPLE_SOURCES, globally normalized as in training; cached."""
    if not cache.exists():
        stats = json.loads((packed / 'stats.json').read_text())
        ends = np.cumsum(list(stats['counts'].values()))   # rows are packed contiguously by source
        starts = dict(zip(stats['counts'], ends - np.array(list(stats['counts'].values()))))
        rng = np.random.default_rng(seed)
        rows = [int(starts[s] + rng.integers(stats['counts'][s])) for s in EXAMPLE_SOURCES]
        x = np.load(packed / 'spectra.npy', mmap_mode='r')[rows].astype(np.float32)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, x=(x - stats['mean']) / stats['std'], rows=rows, sources=list(EXAMPLE_SOURCES))
    d = np.load(cache)
    return d['x'], [str(s) for s in d['sources']]


def input_figure(x, sources, name='fig0_input_patches_mask.svg', num_patches=48, mask_ratio=.75, seed=0):
    """Rows: example spectra. Columns: the input, its contiguous patches, and one random 75% mask."""
    import torch
    from spectral_lejepa.models.masking import random_mask
    from spectral_lejepa.models.tokenizer import patch_bounds
    bounds = patch_bounds(x.shape[1], num_patches)
    masked, _ = random_mask(len(x), num_patches, mask_ratio, torch.Generator().manual_seed(seed))
    masked = [set(m.tolist()) for m in masked]
    navy, teal, orange, grey = '#1f4e79', '#2a9d8f', '#d9822b', '#b8c2c6'
    fig, axes = plt.subplots(len(x), 3, figsize=(11, 1.9 * len(x) + .5), sharex='col', squeeze=False)
    for i, (sig, src) in enumerate(zip(x, sources)):
        a, b, c = axes[i]
        a.plot(np.arange(len(sig)), sig, color=navy, lw=1)
        a.set_ylabel(src.replace('labeled_regression/', ''), fontsize=8)
        for p, (lo, hi) in enumerate(bounds):
            seg = np.arange(lo, min(hi + 1, len(sig)))   # overlap one bin so the line is continuous
            b.axvspan(lo - .5, hi - .5, color='#e6eef0' if p % 2 else 'white', lw=0)
            b.plot(seg, sig[seg], color=orange if p % 2 else navy, lw=1.2)
            hidden = p in masked[i]
            if hidden:
                c.axvspan(lo - .5, hi - .5, color='#fbe3e0', lw=0)
            c.plot(seg, sig[seg], color=grey if hidden else teal, lw=1, ls=':' if hidden else '-')
        for ax in (b, c):
            ax.set_xticks([(lo + hi - 1) / 2 for lo, hi in bounds[::8]], [f'p{p}' for p in range(0, num_patches, 8)])
        b.set_ylim(a.get_ylim()), c.set_ylim(a.get_ylim())
    n_mask = len(masked[0])
    titles = ('(a) input spectrum, 245 bins (normalized)', f'(b) {num_patches} contiguous patches',
              f'(c) mask: {num_patches - n_mask} visible, {n_mask} masked (shaded)')
    for ax, t in zip(axes[0], titles):
        ax.set_title(t, fontsize=9)
    axes[-1, 0].set_xlabel('bin')
    axes[-1, 1].set_xlabel('patch'), axes[-1, 2].set_xlabel('patch')
    save(fig, name)
    return {'widths': [hi - lo for lo, hi in bounds], 'masked': [sorted(m) for m in masked]}


BENCHMARK = Path('outputs/benchmark/results.json')
BENCH_ARMS = (('raw', 'Raw spectrum', '#bdbdbd'), ('data2vec_feb25', 'data2vec (Feb 25)', '#9e9ac8'),
              ('data2vec', 'data2vec (recent, all data)', '#6a51a3'), ('jepa', 'JEPA ensemble', '#1f4e79'),
              ('pred_avg', 'JEPA + MiniRocket, mean prediction', '#2a9d8f'), ('minirocket_s0', 'MiniRocket', '#d9822b'))


def benchmark_figure(res, name='fig_benchmark.svg'):
    """Nested-CV R² per set: raw, both data2vec checkpoints, the JEPA ensemble, JEPA + MiniRocket, MiniRocket."""
    sets = list(res['sets'])
    arms = [a for a in BENCH_ARMS if all(a[0] in res['sets'][s]['r2'] for s in sets)]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    w = .8 / len(arms)
    x = np.arange(len(sets))
    for i, (key, label, color) in enumerate(arms):
        ax.bar(x + (i - (len(arms) - 1) / 2) * w, [res['sets'][s]['r2'][key] for s in sets], w,
               yerr=[res['sets'][s]['vs_raw'].get(key, {}).get('sd', 0) for s in sets],
               color=color, label=label, error_kw={'lw': .6, 'ecolor': '#555'})
    ax.set_xticks(x, [SHORT.get(s, s).replace('dataset', '') for s in sets])
    ax.set_ylabel('nested-CV R²')
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc='upper left', bbox_to_anchor=(0, 1.25))
    save(fig, name)


def ablation_stats(root=ABLATION):
    """Per job: design (depth, λ, seed), per-block flat-readout mean ΔR² vs raw (8 sets) and pooled isotropy."""
    jobs = {}
    for d in sorted(p for p in Path(root).iterdir() if (p / 'isotropy.json').exists()):
        depth, lam, seed = json.loads((d / 'state.json').read_text())['design']
        run = sorted((d / 'eval').glob('*_step*'))[-1]
        flat = []
        for k in range(depth + 1):
            gains = []
            for ds in DATASETS:
                fam = json.loads((run / ds / 'nested_results.json').read_text())['families']
                gains.append(fam[f'layer{k}/flat']['r2_mean'] - fam['raw']['r2_mean'])
            flat.append(float(np.mean(gains)))
        iso = json.loads((d / 'isotropy.json').read_text())['blocks']
        jobs[d.name] = {'depth': depth, 'lam': lam, 'seed': seed, 'flat': flat,
                        **{m: [iso[str(k)]['trained'][m] for k in range(depth + 1)]
                           for m in ('effective_rank', 'mean_cosine', 'sigreg')}}
    return jobs


def ablation_criteria(jobs):
    """Criteria registered before the runs (local spec 2026-10-14-sigreg-depth-ablation-design)."""
    lam = {j['lam']: j for j in jobs.values() if j['depth'] == 6 and j['seed'] == 0}
    ref, off = lam[.05], lam[0.]
    ranks = [lam[x]['effective_rank'][6] for x in sorted(lam)]
    by = {(j['depth'], j['seed']): j for j in jobs.values() if j['lam'] == .05}
    return {'S1': off['effective_rank'][6] < .1 * ref['effective_rank'][6] and off['flat'][2] < ref['flat'][2],
            'S2': all(a < b for a, b in zip(ranks, ranks[1:])),
            'D1': all(by[(6, s)]['flat'][2] > by[(2, s)]['flat'][2] for s in (0, 1))}


def ablations(root=ABLATION):
    jobs = ablation_stats(root)
    lam = sorted((j for j in jobs.values() if j['depth'] == 6 and j['seed'] == 0), key=lambda j: j['lam'])
    x = np.arange(len(lam))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for k, color in ((2, '#c0504d'), (6, '#4f81bd')):
        axes[0].plot(x, [j['effective_rank'][k] for j in lam], marker='o', color=color, label=f'block {k}')
        axes[1].plot(x, [j['mean_cosine'][k] for j in lam], marker='o', color=color, label=f'block {k}')
    axes[2].bar(x, [j['flat'][2] for j in lam], color=['#888' if j['lam'] == 0 else '#c0504d' for j in lam])
    axes[2].axhline(0, color='k', lw=.6)
    for ax, title in zip(axes, ('pooled effective rank', 'mean pairwise cosine (pooled)',
                                'block-2 flat: mean ΔR² vs raw (8 sets)')):
        ax.set(title=title, xlabel='SIGReg weight λ', xticks=x, xticklabels=[f"{j['lam']:g}" for j in lam])
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, 'fig9_sigreg_ablation.svg')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    colors = {2: '#9bbb59', 4: '#f79646', 6: '#c0504d'}
    for j in sorted((j for j in jobs.values() if j['lam'] == .05), key=lambda j: (j['depth'], j['seed'])):
        kw = dict(color=colors[j['depth']], marker='o', ms=3, ls='-' if j['seed'] == 0 else '--',
                  label=f"depth {j['depth']}, seed {j['seed']}")
        blocks = np.arange(j['depth'] + 1)
        axes[0].plot(blocks, np.maximum(j['flat'], -.25), **kw)
        axes[1].plot(blocks, j['effective_rank'], **kw)
    axes[0].axhline(0, color='k', lw=.6)
    axes[0].axvline(2, color='k', lw=.6, ls=':')
    axes[0].set(title='flat readout: mean ΔR² vs raw (8 sets; clipped at −.25)', xlabel='encoder block', ylim=(-.26, .12))
    axes[1].set(title='pooled effective rank', xlabel='encoder block')
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, 'fig10_depth_ablation.svg')
    return {'jobs': jobs, 'criteria': ablation_criteria(jobs)}


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


def figures():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'fig0_architecture.svg').write_text(block_diagram())
    input_figure(*input_examples())
    hist = json.loads(HIST.read_text())
    res = json.loads((ROOT / 'results_analysis.json').read_text())
    pretraining_curves(hist)
    continuation_curves(hist)
    candidates()
    ensemble_scores()
    true_vs_predicted(ensemble_oof())
    member_heatmap(res)
    robustness(res)
    depth_profile()
    if BENCHMARK.exists():
        benchmark_figure(json.loads(BENCHMARK.read_text()))
    print(json.dumps(ablations()["criteria"]))


def main():
    fetch() if sys.argv[1:] == ['fetch'] else figures()


if __name__ == '__main__':
    main()
