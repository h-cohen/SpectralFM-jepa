"""Training-free time-series baselines (ROCKET family, sktime) against the JEPA ensemble, same folds.

    python -m spectral_lejepa benchmark run --arms minirocket_s0 rocket ... [--sets ...] [--device cuda]
    python -m spectral_lejepa benchmark score [--output outputs/benchmark/results.json]

`run` reads each set's rows from a member's bank.npz (input_raw, y), builds the arm's features and
runs the members' nested CV (2x5 outer, 5 inner, 12 recipes, seed 42); out-of-fold predictions go to
<out>/<set>/<arm>.npz and finished arms are skipped. Arms: rocket, minirocket_s<seed>, multirocket,
concat_<member> ([member block-2 flat | MiniRocket seed 0]). `score` adds two combinations, pred_avg
(mean of the JEPA ensemble and MiniRocket seed 0 predictions) and concat (mean of the members' concat
arms), and checks the criteria registered before the runs (B1-B3).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from spectral_lejepa.evaluation.ensemble import ensemble_predictions
from spectral_lejepa.evaluation.nested import nested_cv, paired_delta
from spectral_lejepa.evaluation.probe import r2

OUT = Path('outputs/benchmark')
MEMBERS = [*sorted(Path('outputs/robust-ensemble').glob('robust_*/eval/*_step*')),
           *Path('outputs/robustness-continuation/robust_s401/eval').glob('*_step*')]
SETS = ('labeled_data', 'dataset0055', 'dataset0106', 'dataset0109', 'dataset0112', 'dataset0113',
        'dataset0114', 'dataset0120')
PRIMARY = 'minirocket_s0'
DESCRIPTIVE = ('rocket', 'minirocket_s1', 'minirocket_s2', 'multirocket')


def features(kind, x, seed=0, n_jobs=8):
    """[N, L] raw spectra -> [N, F] sktime features. Unsupervised: fit() sees only x (for MiniRocket and
    MultiRocket, the bias quantiles), never labels."""
    from sktime.transformations.panel.rocket import MiniRocket, MultiRocket, Rocket
    make = {'rocket': lambda: Rocket(num_kernels=10000, random_state=seed, n_jobs=n_jobs),
            'minirocket': lambda: MiniRocket(random_state=seed, n_jobs=n_jobs),
            'multirocket': lambda: MultiRocket(random_state=seed, n_jobs=n_jobs)}[kind]
    X = np.asarray(x, dtype=np.float64)[:, None, :]
    return np.asarray(make().fit_transform(X), dtype=np.float32)


def _member_name(bank_dir):
    return Path(bank_dir).parents[1].name


def arm_features(arm, ds, members=MEMBERS):
    bank = np.load(Path(members[0]) / ds / 'bank.npz', allow_pickle=True)
    raw, y = bank['input_raw'][:, 0], bank['y']
    if arm.startswith('concat_'):
        (m,) = [b for b in members if _member_name(b) == arm[len('concat_'):]]
        mb = np.load(Path(m) / ds / 'bank.npz', allow_pickle=True)
        if not np.array_equal(mb['y'], y):
            raise ValueError(f'{ds}: {arm} rows differ from {members[0]}')
        return np.hstack([mb['bank__layer2__flat'][:, 0, 0], features('minirocket', raw, 0)]), y
    kind, _, seed = arm.partition('_s')
    return features(kind, raw, int(seed or 0)), y


def run(arms, sets=SETS, out=OUT, device='cuda', members=MEMBERS):
    for ds in sets:
        for arm in arms:
            path = Path(out) / ds / f'{arm}.npz'
            if path.exists():
                continue
            X, y = arm_features(arm, ds, members)
            res = nested_cv({arm: X}, {arm: [arm]}, y, seed=42, backend='torch', device=device)[arm]
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, y=y, oof=res['oof'], dim=X.shape[1],
                                chosen=json.dumps(res['chosen']))
            print(f'{ds:13s} {arm:22s} dim {X.shape[1]:6d} R² {res["r2_mean"]:.3f}', flush=True)


def _r2(y, P):
    return float(np.mean([r2(y, p) for p in P]))


def score(out=OUT, members=MEMBERS):
    out, names = Path(out), [_member_name(m) for m in members]
    res = {'members': [str(m) for m in members], 'primary': PRIMARY, 'sets': {}}
    for d in (out / s for s in SETS if (out / s).is_dir()):
        oofs = [np.load(Path(m) / d.name / 'nested_oof.npz') for m in members]
        y = oofs[0]['y']
        P = {'raw': oofs[0]['raw'], 'jepa': ensemble_predictions(oofs)}
        for f in d.glob('*.npz'):
            a = np.load(f)
            if not np.array_equal(a['y'], y):
                raise ValueError(f'{f}: rows differ from the members')
            P[f.stem] = a['oof']
        if PRIMARY in P:
            P['pred_avg'] = (P['jepa'] + P[PRIMARY]) / 2
        if all(f'concat_{n}' in P for n in names):
            P['concat'] = np.mean([P[f'concat_{n}'] for n in names], 0)
        res['sets'][d.name] = {'n': int(len(y)), 'r2': {k: _r2(y, p) for k, p in P.items()},
                               'vs_raw': {k: paired_delta(y, p, P['raw']) for k, p in P.items() if k != 'raw'},
                               'vs_jepa': {k: paired_delta(y, p, P['jepa']) for k, p in P.items()
                                           if k not in ('raw', 'jepa')}}
    S = res['sets'].values()
    def wins(a, b):
        return int(sum(s['r2'][a] > s['r2'][b] for s in S if a in s['r2'] and b in s['r2']))
    res['criteria'] = {
        'B1': {'jepa_beats_minirocket': wins('jepa', PRIMARY), 'pass': wins('jepa', PRIMARY) >= 5},
        'B2': {'pred_avg_beats_both': int(sum(s['r2'].get('pred_avg', -9) > max(s['r2']['jepa'], s['r2'].get(PRIMARY, 9))
                                               for s in S)), 'pass': None},
        'B3': {'concat_beats_jepa': wins('concat', 'jepa'), 'pass': wins('concat', 'jepa') >= 5}}
    res['criteria']['B2']['pass'] = res['criteria']['B2']['pred_avg_beats_both'] >= 5
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(prog='spectral_lejepa benchmark')
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--arms', nargs='+', required=True)
    r.add_argument('--sets', nargs='+', default=SETS)
    r.add_argument('--device', default='cuda')
    r.add_argument('--out', default=str(OUT))
    s = sub.add_parser('score')
    s.add_argument('--out', default=str(OUT))
    s.add_argument('--output', default=str(OUT / 'results.json'))
    a = ap.parse_args(argv)
    if a.cmd == 'run':
        run(a.arms, a.sets, a.out, a.device)
    else:
        res = score(a.out)
        Path(a.output).write_text(json.dumps(res, indent=1))
        print(json.dumps(res['criteria']))


if __name__ == '__main__':
    main()
