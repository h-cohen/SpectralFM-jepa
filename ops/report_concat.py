"""Numbers for the component-concatenation study (spec docs/superpowers/specs/2026-11-16-component-concat-design.md),
from the files `evaluate --components 0,1 --concat --minirocket` wrote for bm100 seeds 107 and 108. No refitting.

    PYTHONPATH=src:. python ops/report_concat.py [--dir outputs/component-concat] [--output summary.json]

Families: raw, random (seed-107 run), MiniRocket seed 0 (identical in both runs; checked), LeJEPA = mean of the two
seeds' predictions, LeJEPA s107 = single encoder. Scores comp0, comp1, concat. Rules fixed before results:
concat helps an arm if concat − comp0 > 0 on ≥ 5/7 sets and mean ΔR² ≥ +.01.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ops.report_rmse import delta, summ

SETS = ['dataset0055', 'dataset0106', 'dataset0109', 'dataset0112', 'dataset0113', 'dataset0114', 'dataset0120']
ARMS = ['raw', 'random', 'minirocket', 'jepa_s107', 'jepa']
SCORES = ['comp0', 'comp1', 'comp2', 'concat']


def load(root, ds, comps='0,1'):
    """Every `<run>_step*` under root that has this set; LeJEPA = mean over runs, jepa_s107 = the first run
    (sorted by name). Raw, random and MiniRocket must be identical across runs (same rows, folds, seeds)."""
    runs = [np.load(f) for f in sorted(Path(root).glob(f'*_step*/{ds}/components_oof.npz'))]
    if not runs:
        raise FileNotFoundError(f'{root}: no runs for {ds}')
    a = runs[0]
    for b in runs[1:]:
        for k in ('y', *(f'{arm}__{s}' for arm in ('raw', 'minirocket') for s in SCORES if f'{arm}__{s}' in a)):
            if not np.allclose(a[k], b[k], atol=1e-5):
                raise RuntimeError(f'{ds}: {k} differs between runs')
    P = {'y': a['y'], 'n_runs': len(runs)}
    for s in SCORES:
        if f'raw__{s}' not in a:
            continue
        P[f'raw__{s}'], P[f'random__{s}'], P[f'minirocket__{s}'] = a[f'raw__{s}'], a[f'random__{s}'], a[f'minirocket__{s}']
        P[f'jepa_s107__{s}'] = a[f'jepa__{s}']
        P[f'jepa__{s}'] = np.mean([r[f'jepa__{s}'] for r in runs], axis=0)
    return P


def helps(d):
    d = np.asarray(d)
    return bool((d > 0).sum() >= 5 and d.mean() >= .01)


def summarize(root, sets=SETS):
    per = {}
    for ds in sets:
        P = load(root, ds)
        y = P['y']
        sc = [s for s in SCORES if f'raw__{s}' in P]
        per[ds] = {'n': int(len(y)), 'n_runs': P['n_runs'],
                   'scores': {a: {s: summ(y, P[f'{a}__{s}']) for s in sc} for a in ARMS},
                   'concat_minus_comp0': {a: delta(y, P[f'{a}__concat'], P[f'{a}__comp0']) for a in ARMS},
                   'jepa_concat_vs': {b: delta(y, P['jepa__concat'], P[f'{b}__concat']) for b in ('raw', 'minirocket')},
                   'jepa_comp0_vs': {b: delta(y, P['jepa__comp0'], P[f'{b}__comp0']) for b in ('raw', 'minirocket')}}
    SETS_ = list(per)
    d = {a: [per[s]['concat_minus_comp0'][a]['d_r2'] for s in SETS_] for a in ARMS}
    vs = {b: [per[s]['jepa_concat_vs'][b]['d_r2'] for s in SETS_] for b in ('raw', 'minirocket')}
    common = [s for s in SCORES if all(s in per[x]['scores']['raw'] for x in SETS_)]
    return {'per_set': per,
            'mean': {a: {s: {k: float(np.mean([per[x]['scores'][a][s][k] for x in SETS_])) for k in ('r2', 'rmse')}
                         for s in common} for a in ARMS},
            'concat_helps': {a: {'mean': float(np.mean(v)), 'wins': int((np.array(v) > 0).sum()), 'helps': helps(v)}
                             for a, v in d.items()},
            'jepa_concat_vs': {b: {'mean': float(np.mean(v)), 'wins': int((np.array(v) > 0).sum())} for b, v in vs.items()}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--dir', type=Path, default=Path('outputs/component-concat'))
    ap.add_argument('--output', type=Path)
    ap.add_argument('--sets', nargs='+', default=SETS)
    a = ap.parse_args(argv)
    out = summarize(a.dir, a.sets)
    (a.output or a.dir / 'summary.json').write_text(json.dumps(out, indent=1))
    for ds, r in out['per_set'].items():
        print(ds[-4:], r['n'], f"runs {r['n_runs']}", ' | '.join(f"{arm} " + ' '.join(f"{s} {v['r2']:.3f}" for s, v in r['scores'][arm].items())
                                          for arm in ARMS))
    print('mean', ' | '.join(f"{arm} " + ' '.join(f"{v['r2']:.3f}/{v['rmse']:.3f}"
                                                  for v in out['mean'][arm].values()) for arm in ARMS))
    print('concat − comp0:', out['concat_helps'])
    print('LeJEPA concat vs:', out['jepa_concat_vs'])


if __name__ == '__main__':
    main()
