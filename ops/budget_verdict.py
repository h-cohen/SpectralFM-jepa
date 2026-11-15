"""Verdict of the measured-budget study (spec docs/superpowers/specs/2026-11-12-measured-budget-design.md), from
one `ensemble --stack --no-stack-heads --mix-groups` fusion of all-data (a = nr_s107/108) and the nine budget
arm families (two seeds each). No refitting.

    PYTHONPATH=src:. python ops/budget_verdict.py --dir outputs/measured-budget/ens \
        --runs outputs/measured-budget --bank-groups outputs/measured-budget/bank_groups.json

Rules (fixed before results, seven measured sets; labeled_data reported separately):
- an arm passes at fraction f if arm − all-data ≥ −.01 on ≥ 5/7 sets and the mean is ≥ −.01;
  f* = the smallest passing f;
- synthetic substitutes if MS passes at an f where M fails;
- the blend helps at f if MX − MS > 0 on ≥ 5/7 and the mean is ≥ +.01;
- borderline (add seeds 109/110) if a mean sits within .005 of its bar.
Seen/unseen R² (descriptive): evaluation spectra whose sample group was / was not in the run's measured groups.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ops.budget_seen import seen_mask, seen_r2
from ops.report_rmse import delta, summ

SEVEN = ('dataset0055', 'dataset0106', 'dataset0109', 'dataset0112', 'dataset0113', 'dataset0114', 'dataset0120')
FRACTIONS = {'100': 1.0, '25': .25, '6': .0625, '1p5': .015}
ARMS = {'M': {1.0: 'bm100', .25: 'bm25', .0625: 'bm6', .015: 'bm1p5'},
        'MS': {1.0: 'a', .25: 'bms25', .0625: 'bms6', .015: 'bms1p5'},
        'MX': {.0625: 'bmx6', .015: 'bmx1p5'}}


def passes(d) -> bool:
    d = np.asarray(d)
    return bool((d >= -.01).sum() >= 5 and d.mean() >= -.01)


def f_star(per_f):
    ok = [f for f, d in per_f.items() if passes(d)]
    return min(ok) if ok else None


def substitutes(m, ms) -> bool:
    return any(passes(ms[f]) and not passes(m[f]) for f in ms if f in m and f < 1.0)   # MS at 1.0 is the reference


def blend_helps(d) -> bool:
    d = np.asarray(d)
    return bool((d > 0).sum() >= 5 and d.mean() >= .01)


def borderline(d, bar=-.01) -> bool:
    return bool(abs(np.mean(d) - bar) <= .005)


def _labels():
    return sorted({lab for arm in ARMS.values() for lab in arm.values()})


def _groups(runs_dir, label):
    """Measured group keys of a label's runs (identical across seeds: the selection seed is fixed)."""
    if label == 'a':
        return None
    files = sorted(Path(runs_dir).glob(f'{label}_s*_*/measured_groups.npy'))
    return np.load(files[0]) if files else None


def summarize(fusion_dir, runs_dir=None, bank_groups=None):
    fusion_dir = Path(fusion_dir)
    fusion = json.loads((fusion_dir / 'fusion.json').read_text())
    keys = json.loads(Path(bank_groups).read_text()) if bank_groups else {}
    per_set = {}
    for ds in fusion['sets']:
        z = np.load(fusion_dir / ds / 'fusion_oof.npz')
        y = z['y']
        P = {'raw': z['raw'], 'random': z['random_avg_a'], **{lab: z[f'avg_{lab}'] for lab in _labels()}}
        row = {'n': int(len(y)), 'arms': {k: summ(y, p) for k, p in P.items()},
               'vs_raw': {lab: delta(y, P[lab], P['raw']) for lab in _labels()},
               'minus_a': {lab: delta(y, P[lab], P['a']) for lab in _labels() if lab != 'a'}}
        if runs_dir and ds in keys:
            row['seen'] = {}
            for lab in _labels():
                g = _groups(runs_dir, lab)
                if g is not None:
                    row['seen'][lab] = seen_r2(y, P[lab], seen_mask(keys[ds], g))
        per_set[ds] = row
    D = {lab: np.array([per_set[s]['minus_a'][lab]['d_r2'] for s in SEVEN]) for lab in _labels() if lab != 'a'}
    D['a'] = np.zeros(len(SEVEN))
    per_f = {arm: {f: D[lab] for f, lab in fs.items()} for arm, fs in ARMS.items()}
    mx_ms = {f: np.array([per_set[s]['minus_a'][ARMS['MX'][f]]['d_r2'] - per_set[s]['minus_a'][ARMS['MS'][f]]['d_r2']
                          for s in SEVEN]) for f in ARMS['MX']}
    return {
        'fusion_dir': str(fusion_dir), 'mix_groups': fusion.get('mix_groups'), 'per_set': per_set,
        'mean_vs_raw': {lab: float(np.mean([per_set[s]['vs_raw'][lab]['d_r2'] for s in SEVEN])) for lab in _labels()},
        'mean_minus_a': {lab: float(d.mean()) for lab, d in D.items()},
        'within_01': {lab: int((d >= -.01).sum()) for lab, d in D.items()},
        'passes': {lab: passes(d) for lab, d in D.items()},
        'borderline': {lab: borderline(d) for lab, d in D.items() if lab != 'a'},
        'f_star': {arm: f_star(v) for arm, v in per_f.items()},
        'synthetic_substitutes': substitutes(per_f['M'], per_f['MS']),
        'mx_minus_ms': {str(f): {'mean': float(d.mean()), 'wins': int((d > 0).sum()), 'helps': blend_helps(d),
                                 'borderline': borderline(d, .01)} for f, d in mx_ms.items()},
        'labeled_data_in_verdict': False}


def fig_budget(out, outdir):
    """fig_budget.svg: mean ΔR² vs raw over the seven sets against the measured fraction, one line per arm."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    col = {'M': '#c0504d', 'MS': '#3a6ea5', 'MX': '#6a9a3a'}
    lab = {'M': 'measured only', 'MS': 'measured + synthetic', 'MX': 'measured × synthetic blend'}
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for arm, fs in ARMS.items():
        f = sorted(k for k in fs if not (arm == 'MS' and k == 1.0))
        ax.plot([100 * x for x in f], [out['mean_vs_raw'][fs[x]] for x in f], 'o-', color=col[arm], label=lab[arm])
    ax.axhline(out['mean_vs_raw']['a'], color='#3a6ea5', ls='--', lw=1, label='all data (nr, 100%)')
    ax.axhline(0, color='k', lw=.6, label='raw')
    ax.set_xscale('log')
    ax.set_xticks([1.5, 6.25, 25, 100], ['1.5%', '6.25%', '25%', '100%'])
    ax.set_xlabel('measured sample groups used in pretraining')
    ax.set_ylabel('mean ΔR² vs raw (7 sets)')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / 'fig_budget.svg')
    plt.close(fig)


def _print(out):
    sets = [s for s in ('labeled_data', *SEVEN) if s in out['per_set']]
    labs = _labels()
    print('set'.ljust(13) + 'raw R²  RMSE    random ' + ' '.join(l.rjust(7) for l in labs))
    for s in sets:
        r = out['per_set'][s]
        a = r['arms']
        print(f"{s:13s}{a['raw']['r2']:.3f}  {a['raw']['rmse']:.4g}  {a['random']['r2']:.3f}  "
              + ' '.join(f"{r['vs_raw'][l]['d_r2']:+.3f}" for l in labs))
    print('mean vs raw  ' + ' '.join(f"{out['mean_vs_raw'][l]:+.3f}" for l in labs))
    print('mean vs all  ' + ' '.join(f"{out['mean_minus_a'][l]:+.3f}" for l in labs))
    print('f*:', out['f_star'], '| synthetic substitutes:', out['synthetic_substitutes'])
    print('MX − MS:', out['mx_minus_ms'])
    print('borderline:', [k for k, v in out['borderline'].items() if v])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--dir', type=Path, required=True)
    ap.add_argument('--runs', type=Path)
    ap.add_argument('--bank-groups', type=Path)
    ap.add_argument('--output', type=Path)
    a = ap.parse_args(argv)
    out = summarize(a.dir, a.runs, a.bank_groups)
    (a.output or a.dir.parent / 'verdict.json').write_text(json.dumps(out, indent=1))
    fig_budget(out, a.dir.parent / 'figures')
    _print(out)


if __name__ == '__main__':
    main()
