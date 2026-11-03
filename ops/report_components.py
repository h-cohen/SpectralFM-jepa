"""Numbers and figures for report Appendix G (second-component study), from the files `evaluate --components 0,1
--pooled` wrote. No refitting.

    PYTHONPATH=src:. python ops/report_components.py [--dir DIR] [--summary outputs/components/summary.json]
                                                    [--figures docs/figures/training-report]

Scores: comp0, comp1 (fit and scored on one component), pooled_comp0 / pooled_comp1 (fit on both components of the
training samples, scored on one), pooled_avg (the two pooled predictions averaged per sample) and pooled_all
(computed here: the pooled fit scored over all 2n test rows together, i.e. pooled_comp0 and pooled_comp1
concatenated against y repeated twice). R² and RMSE are per repeat, then averaged over the two repeats, as in
nested.py; RMSE is in label units (labels are z-scored per set). Paired deltas vs comp0 use 1,000 bootstrap
resamples of samples (seed 0); for pooled_all both rows of a resampled sample stay together. The script checks
every R² and RMSE it can recompute against results_components.json and prints the largest gap.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from spectral_lejepa.evaluation.probe import r2

DEFAULT_DIR = ('/mnt5/home/hadar/nova/SpectralFM-jepa/outputs/components/'
               'nr_s107_20261022-204051_step497990')
SETS = ['0055', '0106', '0109', '0112', '0113', '0114', '0120']
ARMS = ['raw', 'random', 'jepa']
ARM_LABEL = {'raw': 'Raw ridge', 'random': 'Random-weight encoder', 'jepa': 'JEPA nr_s107'}
SCORES = ['comp0', 'comp1', 'pooled_comp0', 'pooled_comp1', 'pooled_all', 'pooled_avg']
SCORE_LABEL = {'comp0': 'comp0', 'comp1': 'comp1', 'pooled_comp0': 'pooled on comp0',
               'pooled_comp1': 'pooled on comp1', 'pooled_all': 'pooled on all rows',
               'pooled_avg': 'pooled, per-sample average'}


def _rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def r2m(y, P, rep=1):
    return float(np.mean([r2(np.tile(y, rep), p) for p in P]))


def rmsem(y, P, rep=1):
    return float(np.mean([_rmse(np.tile(y, rep), p) for p in P]))


def preds(z, arm, score):
    """[R, n*k] predictions and the row repeat k (2 for pooled_all)."""
    if score == 'pooled_all':
        return np.concatenate([z[f'{arm}__pooled_comp0'], z[f'{arm}__pooled_comp1']], axis=1), 2
    return z[f'{arm}__{score}'], 1


def resampled(y, P, k, i):
    """Predictions and labels for resampled samples i; k=2: both rows of a sample stay together."""
    n = len(y)
    if k == 1:
        return y[i], P[:, i]
    return np.tile(y[i], 2), np.concatenate([P[:, :n][:, i], P[:, n:][:, i]], axis=1)


def metrics(y, P, k):
    n = len(y)
    yy = np.tile(y, k)
    return {'r2': float(np.mean([r2(yy, p) for p in P])), 'rmse': float(np.mean([_rmse(yy, p) for p in P]))}


def delta_vs_comp0(y, P, k, P0, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    d_r2, d_rm = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        yb, Pb = resampled(y, P, k, i)
        y0, P0b = resampled(y, P0, 1, i)
        d_r2.append(np.mean([r2(yb, a) for a in Pb]) - np.mean([r2(y0, b) for b in P0b]))
        d_rm.append(np.mean([_rmse(yb, a) for a in Pb]) - np.mean([_rmse(y0, b) for b in P0b]))
    m, m0 = metrics(y, P, k), metrics(y, P0, 1)
    return {'d_r2': m['r2'] - m0['r2'], 'sd_r2': float(np.std(d_r2)),
            'd_rmse': m['rmse'] - m0['rmse'], 'sd_rmse': float(np.std(d_rm))}


def collect(root):
    out, gap = {}, 0.0
    for s in SETS:
        res = json.load(open(root / f'dataset{s}' / 'results_components.json'))
        z = np.load(root / f'dataset{s}' / 'components_oof.npz')
        y = np.asarray(z['y'], float)
        row = {'n': res['protocol']['n'], 'arms': {}}
        for a in ARMS:
            P0 = z[f'{a}__comp0']
            row['arms'][a] = {}
            for sc in SCORES:
                P, k = preds(z, a, sc)
                m = metrics(y, P, k)
                if sc != 'pooled_all':   # check against the stored numbers
                    st = res['scores'][a][sc]
                    gap = max(gap, abs(m['r2'] - st['r2_mean']), abs(m['rmse'] - st['rmse_mean']))
                m['bootstrap_sd'] = None
                if sc == 'comp0':
                    m.update({'d_r2': 0.0, 'sd_r2': 0.0, 'd_rmse': 0.0, 'sd_rmse': 0.0})
                else:
                    m.update(delta_vs_comp0(y, P, k, P0))
                    if sc != 'pooled_all':
                        sd = res['deltas_vs_comp0'][a][sc]
                        gap = max(gap, abs(m['d_r2'] - sd['r2']['delta']), abs(m['d_rmse'] - sd['rmse']['delta']))
                row['arms'][a][sc] = m
        out[s] = row
    return out, gap


def summary(per_set):
    """Per arm x score: wins vs comp0 (sets with dR2 > 0), mean dR2 and dRMSE; SD = RMS of per-set bootstrap SDs / sqrt(7)
    (the mean of seven independent per-set deltas)."""
    tab = {}
    for a in ARMS:
        tab[a] = {}
        for sc in SCORES:
            d = [per_set[s]['arms'][a][sc] for s in SETS]
            tab[a][sc] = {'mean_r2': float(np.mean([x['r2'] for x in d])),
                          'mean_rmse': float(np.mean([x['rmse'] for x in d])),
                          'wins': int(sum(x['d_r2'] > 0 for x in d)),
                          'mean_d_r2': float(np.mean([x['d_r2'] for x in d])),
                          'sd_d_r2': float(np.sqrt(np.sum([x['sd_r2'] ** 2 for x in d])) / len(d)),
                          'mean_d_rmse': float(np.mean([x['d_rmse'] for x in d])),
                          'sd_d_rmse': float(np.sqrt(np.sum([x['sd_rmse'] ** 2 for x in d])) / len(d))}
    return tab


def figures(per_set, tab, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    outdir.mkdir(parents=True, exist_ok=True)
    cols = ['#999999', '#bbbbbb', '#3a6ea5', '#7fa6cf', '#c0504d', '#4f8a4f']
    fig, ax = plt.subplots(figsize=(8, 3.8))
    w = .13
    for j, sc in enumerate(SCORES):
        v = [per_set[s]['arms']['jepa'][sc]['r2'] for s in SETS]
        ax.bar(np.arange(len(SETS)) + (j - 2.5) * w, v, w, color=cols[j], label=SCORE_LABEL[sc])
    ax.set_xticks(range(len(SETS)), SETS)
    ax.set_ylabel('R² (JEPA nr_s107)')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, ncol=3, loc='lower center')
    fig.tight_layout()
    fig.savefig(outdir / 'fig_components_r2.svg')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    sc_ = SCORES[1:]
    acol = {'raw': '#999999', 'random': '#3a6ea5', 'jepa': '#c0504d'}
    for j, a in enumerate(ARMS):
        v = [tab[a][sc]['mean_d_r2'] for sc in sc_]
        e = [tab[a][sc]['sd_d_r2'] for sc in sc_]
        ax.bar(np.arange(len(sc_)) + (j - 1) * .26, v, .26, yerr=e, capsize=2, color=acol[a], label=ARM_LABEL[a])
    ax.axhline(0, color='k', lw=.6)
    ax.set_xticks(range(len(sc_)), [SCORE_LABEL[s] for s in sc_], fontsize=7)
    ax.set_ylabel('mean ΔR² vs comp0 (± SD)')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / 'fig_components_delta.svg')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--dir', default=DEFAULT_DIR)
    p.add_argument('--summary', default='outputs/components/summary.json')
    p.add_argument('--figures', default='docs/figures/training-report')
    a = p.parse_args()
    per_set, gap = collect(Path(a.dir))
    tab = summary(per_set)
    print(f'largest gap to results_components.json (R², RMSE, deltas): {gap:.2e}')
    Path(a.summary).parent.mkdir(parents=True, exist_ok=True)
    json.dump({'per_set': per_set, 'summary': tab, 'max_gap_vs_json': gap}, open(a.summary, 'w'), indent=1)
    figures(per_set, tab, Path(a.figures))
    for arm in ARMS:
        for sc in SCORES:
            t = tab[arm][sc]
            print(f"{arm:7s} {sc:13s} R2 {t['mean_r2']:.3f} RMSE {t['mean_rmse']:.3f} wins {t['wins']}/7 "
                  f"dR2 {t['mean_d_r2']:+.3f}±{t['sd_d_r2']:.3f} dRMSE {t['mean_d_rmse']:+.3f}±{t['sd_d_rmse']:.3f}")


if __name__ == '__main__':
    main()
