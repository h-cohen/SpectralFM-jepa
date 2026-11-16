"""Which ridge penalty λ the nested-CV probe actually chose (report Appendix H), for the measured-only 100% encoder
(Appendix J `bm100`, seeds 107 and 108), from its saved evaluation. No new choice is made: for every outer fold the
recorded winning recipe (normalizer, ridge or plain least squares) is refit on the fold's training rows, which gives
the λ RidgeCV picked by leave-one-out error and the whole LOO error curve over its 20 values (0.001–1000). Each refit
must reproduce the stored out-of-fold prediction, or the script stops.

    PYTHONPATH=src:. python ops/ridge_lambda.py [--eval DIR] [--out outputs/ridge-lambda] [--figures DIR]

Arms: raw spectra, LeJEPA block-2 flat, random-weight block-2 flat. Writes lambdas.json (one record per run, set,
arm, repeat and fold), summary.json and the figures fig_ridge_lambda.svg and fig_ridge_lambda_curves.svg.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from spectral_lejepa.evaluation import probe_torch
from spectral_lejepa.evaluation.bank import load_bank

EVAL = Path('/mnt5/home/hadar/nova/SpectralFM-jepa-budget/outputs/measured-budget/eval')
RUNS = ('bm100_s107', 'bm100_s108')
SETS = ('labeled_data', 'dataset0055', 'dataset0106', 'dataset0109', 'dataset0112', 'dataset0113', 'dataset0114',
        'dataset0120')
ARMS = {'raw': ('', 'raw'), 'jepa': ('', 'layer2/flat'), 'random': ('random_control', 'layer2/flat')}
ALPHAS = probe_torch.ALPHAS


def fold_records(set_dir, arm, sub, family, device):
    res = json.loads((set_dir / sub / 'nested_results.json').read_text())
    oof = np.load(set_dir / sub / 'nested_oof.npz')
    bank, raw, y, _ = load_bank(set_dir / sub / 'bank.npz', readouts=('flat',))
    X = np.asarray(raw if family == 'raw' else bank[family], np.float32)
    seed0 = res['protocol']['seed']
    out = []
    for c in res['families'][family]['chosen']:
        r, f = c['repeat'], c['fold']
        te = np.array(res['test_folds'][r][f])
        tr = np.setdiff1d(np.arange(len(y)), te)
        seed = seed0 + 1000 + 10 * r + f
        Z_tr, Z_te = probe_torch.fold_transforms(X[tr], X[te], seed=seed, device=device)[c['norm']]
        fit = probe_torch.ridge_gcv(Z_tr, y[tr])
        rec = {'arm': arm, 'repeat': r, 'fold': f, 'n_train': int(len(tr)), 'dim': int(X.shape[1]),
               'norm': c['norm'], 'probe': c['probe'], 'inner_r2': c['inner_r2'],
               'alpha': fit['alpha'] if c['probe'] == 'ridgecv' else None,
               'loo_mse': [float(v) for v in fit['loo_mse']]}
        if c['probe'] == 'ridgecv':
            pred = (Z_te.to(torch.float64) @ fit['coef'] + fit['intercept']).cpu().numpy()
            gap = float(np.max(np.abs(pred - oof[family][r, te])))
            if gap > 1e-4 * float(np.std(y)):
                raise RuntimeError(f'{set_dir.name} {arm} r{r} f{f}: refit differs from stored OOF by {gap:.2e}')
            rec['refit_gap'] = gap
        out.append(rec)
    return out


def summarize(records):
    """Per set and arm: folds won by ridge vs plain LS, and the chosen λ (median, range, share at a grid edge)."""
    out = {}
    for ds in SETS:
        out[ds] = {}
        for arm in ARMS:
            R = [r for r in records if r['set'] == ds and r['arm'] == arm]
            a = np.array([r['alpha'] for r in R if r['alpha'] is not None])
            out[ds][arm] = {'folds': len(R), 'ridge': int(len(a)), 'ols': int(len(R) - len(a)),
                            'n_train': int(np.median([r['n_train'] for r in R])),
                            'norms': {k: sum(r['norm'] == k for r in R) for k in sorted({r['norm'] for r in R})},
                            'alpha_median': float(np.median(a)) if len(a) else None,
                            'alpha_min': float(a.min()) if len(a) else None,
                            'alpha_max': float(a.max()) if len(a) else None,
                            'at_low_edge': int((a == ALPHAS[0]).sum()), 'at_high_edge': int((a == ALPHAS[-1]).sum())}
    return out


def figures(records, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    col = {'raw': '#7f7f7f', 'jepa': '#3a6ea5', 'random': '#c0504d'}
    lab = {'raw': 'raw', 'jepa': 'LeJEPA', 'random': 'random weights'}
    # chosen λ per fold and set; plain least squares drawn at the right of the grid
    fig, ax = plt.subplots(figsize=(8, 4.2))
    rng = np.random.default_rng(0)
    ols_x = np.log10(ALPHAS[-1]) + 1.2
    for i, ds in enumerate(SETS):
        for j, arm in enumerate(ARMS):
            R = [r for r in records if r['set'] == ds and r['arm'] == arm]
            x = np.array([np.log10(r['alpha']) if r['alpha'] is not None else ols_x for r in R])
            yy = i + (j - 1) * .25 + rng.uniform(-.06, .06, len(x))
            ax.scatter(x, yy, s=14, color=col[arm], alpha=.75, label=lab[arm] if i == 0 else None, lw=0)
    ax.axvline(np.log10(ALPHAS[-1]) + .6, color='k', lw=.6)
    ax.set_yticks(range(len(SETS)), [s.replace('dataset', '') for s in SETS])
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3, ols_x], ['.001', '.01', '.1', '1', '10', '100', '1000', 'plain LS'])
    ax.set_xlabel('ridge penalty λ chosen in each outer fold (2 seeds × 2 repeats × 5 folds)')
    ax.invert_yaxis()
    ax.legend(fontsize=7, loc='lower left')
    fig.tight_layout()
    fig.savefig(outdir / 'fig_ridge_lambda.svg')
    plt.close(fig)
    # LOO error curves of the LeJEPA arm, one panel per set, one line per fold (seed 107), chosen λ marked
    fig, axs = plt.subplots(2, 4, figsize=(10, 4.8), sharex=True)
    for ax, ds in zip(axs.ravel(), SETS):
        R = [r for r in records if r['set'] == ds and r['arm'] == 'jepa' and r['run'] == RUNS[0]]
        for r in R:
            m = np.array(r['loo_mse'])
            ax.plot(np.log10(ALPHAS), m / m.min(), color=col['jepa'], lw=.8, alpha=.6)
            if r['alpha'] is not None:
                ax.plot(np.log10(r['alpha']), 1, 'o', ms=3, color='k')
        ax.set_title(f"{ds.replace('dataset', '')} ({R[0]['norm'] if len(set(x['norm'] for x in R)) == 1 else 'mixed'})",
                     fontsize=8)
        ax.set_ylim(.98, 1.6)
    for ax in axs[1]:
        ax.set_xlabel('log10 λ', fontsize=8)
    for ax in axs[:, 0]:
        ax.set_ylabel('LOO MSE / min', fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / 'fig_ridge_lambda_curves.svg')
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--eval', type=Path, default=EVAL)
    ap.add_argument('--out', type=Path, default=Path('outputs/ridge-lambda'))
    ap.add_argument('--figures', type=Path)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(argv)
    records = []
    for run in RUNS:
        run_dir = next(a.eval.glob(f'{run}_*_step*'))
        for ds in SETS:
            for arm, (sub, family) in ARMS.items():
                for rec in fold_records(run_dir / ds, arm, sub, family, a.device):
                    records.append({'run': run, 'set': ds, **rec})
            print(ds, run, 'done', flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'lambdas.json').write_text(json.dumps(records))
    summ = summarize(records)
    (a.out / 'summary.json').write_text(json.dumps(summ, indent=1))
    figures(records, a.figures or a.out)
    for ds, by in summ.items():
        print(ds, {arm: (v['ridge'], v['ols'], v['alpha_median'], v['alpha_min'], v['alpha_max'], v['at_high_edge'])
                   for arm, v in by.items()})
    print('largest refit gap', max(r.get('refit_gap', 0) for r in records))


if __name__ == '__main__':
    main()
